import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


def make_color_list(cmap, nbins):
    dc = 1 / (nbins -1)
    frac = np.arange(0, 1+dc, dc)
    return [cmap(i) for i in frac]


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    cmap = plt.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.25, .9)

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    lines = ["MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    plot_quantities = ['med', 'per25', 'per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    delta_m = 0.5
    min_m = 10.5
    nbins_m = 1
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    bin_label = '10.5-11.0'

    colors = make_color_list(cmap, len(lines))

    normal_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_uvb_median_ew_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
     
        else:

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for pq in plot_quantities:
                plot_data[f'{bin_label}_{pq}'] = np.zeros(len(fr200))

            no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_ew_{line}.h5')
            with_uvb_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')

            mask = (mass_long > mass_bins[0]) & (mass_long < mass_bins[1])

            for j in range(len(fr200)):
                        
                ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                        
                plot_data[f'{bin_label}_med'][j] = np.nanmedian(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]))
                plot_data[f'{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 25.)
                plot_data[f'{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 75)

            write_dict_to_h5(plot_data, profile_file)

        plt.plot(plot_data['fr200'], plot_data[f'{bin_label}_med'], ls='-', c=colors[l], lw=1.5, label=plot_lines[l])
        if line == 'OVI1031':
            plt.fill_between(plot_data['fr200'], plot_data[f'{bin_label}_per75'], plot_data[f'{bin_label}_per25'], 
                               alpha=0.3, color=colors[l])

    plt.ylim(-2., 0.5)
    plt.axhline(0., c='k', ls='--', lw=1)

    plt.ylabel(r'${\rm log }( {\rm EW}_{\rm Collisional} / {\rm EW}_{\rm Collisional + UVB} )$')
    plt.xlabel(r'$\rho / r_{200}$')
    plt.legend(loc=4)

    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_ew_profile_mstar.png')
    plt.show()
    plt.close()
