import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$',
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['med', 'per25', 'per75']
    norients = 8
    fr200 = 1.0
    log_frad = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0']
    log_frad = ['0.0', '0.5', '1.0', '1.5', '2.0']

    delta_n = 5
    min_n = 0
    nbins_n = 5
    nsat_bins = np.arange(min_n, min_n+(nbins_n)*delta_n, delta_n)
    nsat_bins[0] = 1.
    xmax = 50.

    colors = make_color_list(plt.get_cmap('viridis'), len(log_frad))

    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    normal_dir = f'/disk04/sapple/data/normal/results/'
    normal_dir = f'/disk04/sapple/data/normal/old_sample/results/'
    results_dir = f'/disk04/sapple/data/satellites/results/'
    sample_dir = f'/disk04/sapple/data/samples/'
    
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf: # change this back to current sample
        nsats_long = np.repeat(sf['nsats'][:], norients)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='row')
    ax = ax.flatten()

    for l, line in enumerate(lines):

        median_file = f'{results_dir}{model}_{wind}_{snap}_{line}_{fr200}r200_sat_median_nsats_ew.h5'
        if os.path.isfile(median_file):
            plot_data = read_h5_into_dict(median_file)

        else:

            normal_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')

            plot_data = {}
            plot_data['nsats'] = get_bin_middle(nsat_bins) 
            plot_data['nsats'] = np.append(plot_data['nsats'], 30.)
            for i in range(len(log_frad)):
                for pq in plot_quantities:
                    plot_data[f'log_frad_{log_frad[i]}_{pq}'] = np.zeros(nbins_n)

            for i in range(len(log_frad)):

                sat_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_{log_frad[i]}log_frad_ew_{line}.h5')
                ew_sat = sat_dict[f'ew_wave_{fr200}r200'].flatten()
                ew_norm = normal_dict[f'ew_wave_{fr200}r200'].flatten()

                for j in range(nbins_n):

                    if j < nbins_n -1:
                        mask = (nsats_long > nsat_bins[j]) & (nsats_long < nsat_bins[j+1])
                    elif j == nbins_n-1:
                        mask = (nsats_long > nsat_bins[j])

                    plot_data[f'log_frad_{log_frad[i]}_med'][j] = np.nanmedian(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]))
                    plot_data[f'log_frad_{log_frad[i]}_per25'][j] = np.nanpercentile(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]), 25.)
                    plot_data[f'log_frad_{log_frad[i]}_per75'][j] = np.nanpercentile(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]), 75.)

            write_dict_to_h5(plot_data, median_file)

        xmax = nsat_bins[1:]
        xmax = np.append(xmax, 50)
        for i in range(len(log_frad)):
            ax[l].hlines(plot_data[f'log_frad_{log_frad[i]}_med'], xmin=nsat_bins,xmax=xmax, color=colors[i], label=r'${{\rm log}} f_{{r_{{\rm half}} \star}} = {{{}}}$'.format(log_frad[i])) 
   
        if l == 0:
            ax[l].legend(fontsize=13, loc=4)
        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$N_{\rm satellites}$')
        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }( {\rm EW}_{\rm sat} / {\rm EW}_{\rm total} )$')
        ax[l].set_ylim(-0.7, 0.)
        ax[l].set_xlim(1, 50)
        ax[l].annotate(plot_lines[l], xy=(0.05, 0.05), xycoords='axes fraction')

        if l == 1:
            ax[l].set_title(r'${\rm Mass\ dependence\ at}\ r_{200}$')
   
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sat_test_{fr200}r200_nsats_median_delta_ew.png')
    plt.show()
    plt.clf()
