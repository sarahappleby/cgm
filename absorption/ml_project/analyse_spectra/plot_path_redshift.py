import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]

    snaps = ['151', '137', '125', '105']
    redshifts = np.array([0., 0.25, 0.5, 1.0])
            
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']

    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}

    norients = 8
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    N_mid = 14.
    N_high = 17.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    idelta = 1. / (len(snaps) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    redshift_colors = [cmap(i) for i in icolor]

    ls = ['--', ':']
    N_labels = [r'$N_{\rm complete} < N < 10^{14} {\rm cm}^{-2}$',
                r'$10^{14} {\rm cm}^{-2} < N < 10^{17} {\rm cm}^{-2}$']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    path_length_file = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/path_lengths.h5'
    if not os.path.isfile(path_length_file):
        create_path_length_file(vel_range, lines, redshift, path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    N_lines = []
    N_lines.append(Line2D([0,1],[0,1], color='dimgrey', ls=ls[0]))
    N_lines.append(Line2D([0,1],[0,1], color='dimgrey', ls=ls[1]))
    leg = ax[0][0].legend(N_lines, N_labels, loc=4, fontsize=14)
    ax[0][0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        path_abs_low = np.zeros(len(snaps))
        path_abs_high = np.zeros(len(snaps))

        for s, snap in enumerate(snaps):
            
            z = redshifts[s]
            with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
                gal_ids = sf['gal_ids'][:]

            nlos = len(gal_ids) * len(fr200) * norients
            dz_total = path_lengths[f'dz_{line}'][list(path_lengths['redshifts']).index(z)] * nlos
            
            chisq_lim = chisq_lim_dict[f'snap_{snap}']
            results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

            all_N = []
            all_chisq = []

            for k in range(len(fr200)):
                with h5py.File(results_file, 'r') as hf:
                    all_N.extend(hf[f'log_N_{fr200[k]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{fr200[k]}r200'][:])
            
            all_N = np.array(all_N)
            all_chisq = np.array(all_chisq)
            low_mask = (all_N > N_min[l]) * (all_N < N_mid) * (all_chisq < chisq_lim[l])
            high_mask = (all_N > N_mid) * (all_N < N_high) * (all_chisq < chisq_lim[l])

            path_abs_low[s] = len(all_N[low_mask]) / dz_total
            path_abs_high[s] = len(all_N[high_mask]) / dz_total

        path_abs_low = np.log10(path_abs_low)
        path_abs_high = np.log10(path_abs_high)

        ax[i][j].plot(np.log10(redshifts+1), path_abs_low, c='dimgrey', ls=ls[0], lw=1)
        ax[i][j].plot(np.log10(redshifts+1), path_abs_high, c='dimgrey', ls=ls[1], lw=1)

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }(1+z)$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log }\ dn/dz$')
            
        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.04, 0.86), xycoords='axes fraction',
                            bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_redshift_path.pdf', format='pdf')
    plt.show()
    plt.close()

