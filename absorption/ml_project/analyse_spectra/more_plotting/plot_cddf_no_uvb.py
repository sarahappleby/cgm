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

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'

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
    snap = sys.argv[3]

    lines = ["MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']
    rho_labels = ['All CGM', 'Inner CGM', 'Outer CGM']
    ssfr_labels = ['All', 'Star forming', 'Green valley', 'Quenched']
    ssfr_colors = ['dimgrey', cb_blue, cb_green, cb_red]
    rho_ls = ['-', '--', ':']
    rho_lw = [1, 1.5, 2]
    logN_min = 11.
    x = [0.77, 0.8, 0.785, 0.785, 0.79]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    fig, ax = plt.subplots(4, 3, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1, 2, 1]}, sharey='row', sharex='col')

    ssfr_lines = []
    for i in range(len(ssfr_colors)):
        ssfr_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[i]))
    leg = ax[0][0].legend(ssfr_lines, ssfr_labels, loc=3, fontsize=12)
    ax[0][0].add_artist(leg)

    #rho_lines = []
    #for i in range(len(rho_ls)):
    #    rho_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[0], ls=rho_ls[i], lw=rho_lw[i]))
    #leg = ax[0][0].legend(rho_lines, rho_labels, loc=1, fontsize=12)
    #ax[0][0].add_artist(leg)

    i = 0
    j = 1

    for l, line in enumerate(lines):

        cddf_file = f'/disk04/sapple/data/collisional/results/{model}_{wind}_{snap}_no_uvb_{line}_cddf_chisqion.h5'

        plot_data = read_h5_into_dict(cddf_file)

        ax[i+1][j].axhline(0, c='k', lw=0.8, ls='-')

        ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_all'], c=ssfr_colors[0], ls=rho_ls[0], lw=1)
        ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_sf'], c=ssfr_colors[1], ls=rho_ls[0], lw=rho_lw[0])
        ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_gv'], c=ssfr_colors[2], ls=rho_ls[0], lw=rho_lw[0])
        ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_q'], c=ssfr_colors[3], ls=rho_ls[0], lw=rho_lw[0])

        ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_sf'] - plot_data[f'cddf_all']),
                        c=ssfr_colors[1], ls=rho_ls[0], lw=rho_lw[0])
        ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_gv'] - plot_data[f'cddf_all']),
                        c=ssfr_colors[2], ls=rho_ls[0], lw=rho_lw[0])
        ax[i+1][j].plot(plot_data['plot_logN'], (plot_data[f'cddf_q'] - plot_data[f'cddf_all']),
                        c=ssfr_colors[3], ls=rho_ls[0], lw=rho_lw[0])

        ax[i][j].set_xlim(logN_min, 18)

        ax[i+1][j].set_xlim(logN_min, 18)

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i+1][j].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
            ax[i+1][j].set_ylabel(r'${\rm CDDF} / {\rm CDDF}_{\rm All}$')
        if line == 'H1215':
            ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.05), xycoords='axes fraction')
        else:
            ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.9), xycoords='axes fraction')

        if line in ['SiIII1206', 'CIV1548']:
            ax[i][j].set_xticks(range(11, 18))
        elif line in ['OVI1031']:
            ax[i][j].set_xticks(range(11, 19))

        j += 1
        if line == 'CII1334':
            i += 2
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_no_uvb_cddf_compressed_chisqion.png')
    plt.show()
    plt.close()

