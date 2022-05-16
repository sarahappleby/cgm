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


def stop_array_after_inf(array):
    mask = np.isinf(array)
    if len(array[mask]) > 0:
        inf_start = np.where(mask)[0][0]
        array[inf_start:] = np.inf
        return array
    else:
        return array


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]

    snaps = ['105', '125', '137', '151']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']

    redshift_labels = [r'$z = 1$', r'$z = 0.5$', r'$z = 0.25$', r'$z = 0$']
    rho_ls = ['-', '--', ':']
    logN_min = 11.
    x = [0.79, 0.74, 0.77, 0.75, 0.755, 0.76]
    ncells = 16

    idelta = 1. / (len(snaps) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.3, .8)
    redshift_colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    fig, ax = plt.subplots(4, 3, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1, 2, 1]}, sharey='row', sharex='col')

    redshift_lines = []
    for i in range(len(redshift_colors)):
        redshift_lines.append(Line2D([0,1],[0,1], color=redshift_colors[i]))
    leg = ax[0][0].legend(redshift_lines, redshift_labels, loc=3, fontsize=14)
    ax[0][0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        redshift_zero_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_151_{line}_cddf_chisqion.h5'
        redshift_zero_data = read_h5_into_dict(redshift_zero_file)

        for s, snap in enumerate(snaps):

            cddf_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'
            plot_data = read_h5_into_dict(cddf_file)

            ax[i+1][j].axhline(0, c='k', lw=1, ls=':')

            if snap is not '151':
                ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_all'] - redshift_zero_data[f'cddf_all']), 
                                c=redshift_colors[s], ls='-', lw=1)
                ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_all'], c=redshift_colors[s], ls=rho_ls[0], lw=1)
            else:
                ax[i][j].errorbar(plot_data['plot_logN'], plot_data[f'cddf_all'], c=redshift_colors[s], yerr=plot_data[f'cddf_all_cv_{ncells}'],
                                  capsize=4, ls=rho_ls[0], lw=1)

            ax[i][j].set_xlim(logN_min, 18)
            ax[i][j].set_ylim(-19, -9)

            ax[i+1][j].set_xlim(logN_min, 18)
            ax[i+1][j].set_ylim(-1.25, 0.75)

            if line in ["SiIII1206", "CIV1548", "OVI1031"]:
                ax[i+1][j].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

            if line in ['H1215', "SiIII1206"]:
                ax[i][j].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
                ax[i+1][j].set_ylabel(r'${\rm CDDF} / {\rm CDDF}_{\rm All}$')
            
            ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.86), xycoords='axes fraction',
                              bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        ax[i+1][j].set_ylim(-1, 1)

        j += 1
        if line == 'CII1334':
            i += 2
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_redshift_cddf_compressed.pdf', format='pdf')
    plt.close()

