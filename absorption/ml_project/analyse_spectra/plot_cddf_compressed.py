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

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '137'

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']
    rho_labels = ['Inner CGM', 'Outer CGM']
    ssfr_labels = ['All', 'Star forming', 'Green valley', 'Quenched']
    ssfr_colors = ['k', cb_blue, cb_green, cb_red]
    rho_ls = ['--', ':']
    logN_min = 11.
    
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey='row', sharex='col')
    ax = ax.flatten()

    ssfr_lines = []
    for i in range(len(ssfr_colors)):
        ssfr_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[i]))
    leg = ax[0].legend(ssfr_lines, ssfr_labels, loc=3, fontsize=12)
    ax[0].add_artist(leg)

    rho_lines = []
    for i in range(len(rho_ls)):
        rho_lines.append(Line2D([0,1],[0,1], color='k', ls=rho_ls[i]))
    leg = ax[0].legend(rho_lines, rho_labels, loc=1, fontsize=12)
    ax[0].add_artist(leg)


    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
        cddf_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_{line}_cddf.h5'

        plot_data = read_h5_into_dict(cddf_file)

        for i in range(len(labels)):

            ax[l].plot(plot_data['plot_logN'], plot_data[f'cddf_all_{labels[i]}'], c=ssfr_colors[0], ls=rho_ls[i], lw=1)
            ax[l].plot(plot_data['plot_logN'], plot_data[f'cddf_sf_{labels[i]}'], c=ssfr_colors[1], ls=rho_ls[i], lw=1)
            ax[l].plot(plot_data['plot_logN'], plot_data[f'cddf_gv_{labels[i]}'], c=ssfr_colors[2], ls=rho_ls[i], lw=1)
            ax[l].plot(plot_data['plot_logN'], plot_data[f'cddf_q_{labels[i]}'], c=ssfr_colors[3], ls=rho_ls[i], lw=1)

        ax[l].set_xlim(logN_min, 18)
        ax[l].set_ylim(-19, -9)

        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')
        ax[l].annotate(plot_lines[l], xy=(0.7, 0.05), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_compressed.png')
    plt.show()
    plt.clf()

