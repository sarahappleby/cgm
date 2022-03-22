import matplotlib.pyplot as plt
from matplotlib import cm
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


def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
    

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'

    line_a = 'H1215'
    line_b = ["CII1334", "CIV1548", "OVI1031"]
    plot_line_a = r'${\rm HI}1215$'
    plot_line_b = [r'${\rm CII}1334$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    #line_a = 'CII1334'
    #line_b = ["CIV1548", "OVI1031"]
    #plot_line_a = r'${\rm CII}1334$'
    #plot_line_b = [r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    redshift = 0.
    quench = quench_thresh(redshift)
    chisq_lim = 2.5
    dv = 25.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    logN_min = 12.

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    fig, ax = plt.subplots(1, len(line_b), figsize=(10, 5), sharey='row', sharex='col')
    #fig, ax = plt.subplots(1, len(line_b), figsize=(8, 5), sharey='row', sharex='col')
    ax = ax.flatten()

    fr200_points = []
    fr200_labels = []
    for i in range(len(fr200)):
        fr200_points.append(plt.scatter([0,1],[0,1],marker='o', color=colors[i]))
        fr200_labels.append(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
    leg = ax[0].legend(fr200_points, fr200_labels, loc=2, fontsize=12)
    ax[0].add_artist(leg, loc=2)

    for l, line in enumerate(line_b):

        for i in range(len(fr200)):

            align_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b[l]}_{fr200[i]}r200.h5'

            with h5py.File(align_file, 'r') as hf:
                all_N_a = hf[f'{line_a}_log_N_{dv}kms'][:]
                all_N_b = hf[f'{line_b[l]}_log_N_{dv}kms'][:]
            mask = (all_N_a > logN_min) & (all_N_b > logN_min) 

            ax[l].scatter(all_N_a[mask], all_N_b[mask], color=colors[i], s=1.5)

    for i in range(len(plot_line_b)):
        ax[i].set_xlim(logN_min, 17)
        ax[i].set_ylim(logN_min, 17)
        ax[i].set_title(plot_line_b[i])
        ax[i].set_xlabel(r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$')

    ax[0].set_ylabel(r'${\rm log }(N\ {\rm Ion} / {\rm cm}^{-2})$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_aligned_column_{line_a}.png')
    plt.show()
    plt.close()


