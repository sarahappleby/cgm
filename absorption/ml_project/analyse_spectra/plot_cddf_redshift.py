import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]


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
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']
    rho_labels = ['All CGM', 'Inner CGM', 'Outer CGM']
    redshift_labels = [r'$z = 1$', r'$z = 0.5$', r'$z = 0.25$', r'$z = 0$']
    redshift_colors = make_color_list(plt.get_cmap('plasma'), len(snaps))
    rho_ls = ['-', '--', ':']
    logN_min = 11.
    x = [0.81, 0.77, 0.8, 0.785, 0.785, 0.79]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    fig, ax = plt.subplots(4, 3, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1, 2, 1]}, sharey='row', sharex='col')

    redshift_lines = []
    for i in range(len(redshift_colors)):
        redshift_lines.append(Line2D([0,1],[0,1], color=redshift_colors[i]))
    leg = ax[0][0].legend(redshift_lines, redshift_labels, loc=3, fontsize=12)
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

            ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_all'], c=redshift_colors[s], ls=rho_ls[0], lw=1)

            if snap is not '151':
                ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_all'] - redshift_zero_data[f'cddf_all']), 
                                c=redshift_colors[s], ls='-', lw=1)

            ax[i][j].set_xlim(logN_min, 18)
            ax[i][j].set_ylim(-19, -9)

            ax[i+1][j].set_xlim(logN_min, 18)
            ax[i+1][j].set_ylim(-0.75, 0.75)

            if line in ["SiIII1206", "CIV1548", "OVI1031"]:
                ax[i+1][j].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

            if line in ['H1215', "SiIII1206"]:
                ax[i][j].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
                ax[i+1][j].set_ylabel(r'${\rm CDDF} / {\rm CDDF}_{\rm All}$')
            if line == 'H1215':
                ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.05), xycoords='axes fraction')
            else:
                ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.9), xycoords='axes fraction')

        j += 1
        if line == 'CII1334':
            i += 2
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_redshift_cddf_compressed.png')
    plt.show()
    plt.close()

