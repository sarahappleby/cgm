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


def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$', 
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['med', 'per25', 'per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    minT = ['4.0', '4.5', '5.0', '5.5']

    delta_m = 0.25
    min_m = 10.
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    
    mass_bin_labels = [] 
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    colors = make_color_list(plt.get_cmap('viridis'), nbins_m)

    results_dir = f'/disk04/sapple/data/collisional/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(len(lines), nbins_m, figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_uvb_median_ew_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
        else:

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for bl in mass_bin_labels:
                for i in range(len(minT)):
                    for pq in plot_quantities:
                        plot_data[f'minT_{minT[i]}_{bl}_{pq}'] = np.zeros(len(fr200))

            for i in range(len(minT)):

                no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_minT_{minT[i]}_ew_{line}.h5')
                with_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_with_uvb_minT_{minT[i]}_ew_{line}.h5')

                for b, bin_label in enumerate(mass_bin_labels):

                    mask = (mass_long > mass_bins[b]) & (mass_long < mass_bins[b+1])

                    for j in range(len(fr200)):
                        
                        ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                        ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                        
                        plot_data[f'minT_{minT[i]}_{bin_label}_med'][j] = np.nanmedian(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]))
                        plot_data[f'minT_{minT[i]}_{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 25.)
                        plot_data[f'minT_{minT[i]}_{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 75)

            write_dict_to_h5(plot_data, profile_file)


        for b, bin_label in enumerate(mass_bin_labels):

            for i in range(len(minT)):

                ax[l][b].plot(plot_data['fr200'], plot_data[f'minT_{minT[i]}_{bin_label}_med'], 
                              ls='-', c=colors[i], label=r'$T_{{\rm min}} = {{{}}}$'.format(minT[i]), lw=1.5)
                if b == 0:
                    ax[l][b].fill_between(plot_data['fr200'], plot_data[f'minT_{minT[i]}_{bin_label}_per75'], plot_data[f'minT_{minT[i]}_{bin_label}_per25'], 
                                          alpha=0.3, color=colors[i])

            ax[l][b].set_ylim(-1.5, 2.0)

            if b == 0:
                ax[l][b].set_ylabel(r'$\Delta {\rm log (EW}/\AA)$')
            if b == nbins_m  -1:
                ax[l][b].annotate(plot_lines[l], xy=(0.05, 0.85), xycoords='axes fraction')
                if l == 0:
                    ax[l][b].legend(loc=1)
            if l == 0:
                ax[l][b].set_title(mass_plot_titles[b])
            if l == len(lines) -1:
                ax[l][b].set_xlabel(r'$\rho / r_{200}$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_ew_profile.png')
    plt.show()
    plt.clf()
