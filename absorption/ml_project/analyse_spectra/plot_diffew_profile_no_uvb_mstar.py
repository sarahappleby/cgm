import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)


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

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    lines = ["MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 01031$']
    plot_quantities = ['med', 'per25', 'per75', 'ndata']
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
    ndata_min = 8

    cmap = plt.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.25, .9)
    colors = make_color_list(cmap, len(lines)+1)[1:]

    normal_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(1, 1)
    
    ion_lines = []
    for i in range(len(lines)):
        ion_lines.append(Line2D([0,1],[0,1], color=colors[i], ls='-', lw=1))
    leg = ax.legend(ion_lines, plot_lines, loc=1, fontsize=12)
    ax.add_artist(leg)

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_uvb_median_ew_profile.h5'

        plot_data = {}
        plot_data['fr200'] = fr200.copy()
        for pq in plot_quantities:
            plot_data[f'{bin_label}_{pq}'] = np.zeros(len(fr200))

        no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_ew_{line}.h5')
        with_uvb_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')

        no_uvb_nlines_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_nlines_{line}.h5')
        with_uvb_nlines_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_nlines_{line}.h5')

        mask = (mass_long > mass_bins[0]) & (mass_long < mass_bins[1])

        print(line)

        for j in range(len(fr200)):
                        
            ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()
            ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()

            nlines_no_uvb = no_uvb_nlines_dict[f'nlines_{fr200[j]}r200'].flatten()
            nlines_with_uvb = with_uvb_nlines_dict[f'nlines_{fr200[j]}r200'].flatten()
            detect_mask = (nlines_no_uvb > 0.) & (nlines_with_uvb > 0.)

            print(f'NLOS: {len(ew_with_uvb[mask*detect_mask])}')

            plot_data[f'{bin_label}_ndata'][j] = len(ew_no_uvb[mask*detect_mask])
            plot_data[f'{bin_label}_med'][j] = np.nanmedian(np.log10(ew_no_uvb[mask*detect_mask]) - np.log10(ew_with_uvb[mask*detect_mask]))
            plot_data[f'{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask*detect_mask]) - np.log10(ew_with_uvb[mask*detect_mask]), 25.)
            plot_data[f'{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask*detect_mask]) - np.log10(ew_with_uvb[mask*detect_mask]), 75)
        
        print('\n')

        write_dict_to_h5(plot_data, profile_file)

        ndata_mask = plot_data[f'{bin_label}_ndata'] > ndata_min
        ax.plot(plot_data['fr200'][ndata_mask], plot_data[f'{bin_label}_med'][ndata_mask], ls='-', c=colors[l], lw=1.5, label=plot_lines[l])
        if False in ndata_mask:
            start = np.where(~ndata_mask)[0][0] - 1
            ax.plot(plot_data['fr200'][start:], plot_data[f'{bin_label}_med'][start:], ls='--', c=colors[l], lw=1.5, label=plot_lines[l]) 

        if line == 'OVI1031':
            ax.fill_between(plot_data['fr200'], plot_data[f'{bin_label}_per75'], plot_data[f'{bin_label}_per25'], 
                               alpha=0.3, color=colors[l])

    ax.set_ylim(-1.5, 0.5)
    ax.axhline(0., c='k', ls='--', lw=1)

    ax.set_ylabel(r'${\rm log }( {\rm EW}_{\rm Collisional} / {\rm EW}_{\rm Collisional + UVB} )$')
    ax.set_xlabel(r'$\rho / r_{200}$')

    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_ew_profile_mstar.png')
    plt.show()
    plt.close()
