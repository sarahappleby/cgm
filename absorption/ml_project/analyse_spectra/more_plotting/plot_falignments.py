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

    line_a = ['H1215', 'H1215', 'H1215', 'CII1334', 'CII1334', "CIV1548"]
    line_b = ["CII1334", "CIV1548", "OVI1031", "CIV1548", "OVI1031", "OVI1031"]
    plot_line_pairs = [r'${\rm HI}1215-{\rm CII}1334$', r'${\rm HI}1215-{\rm CIV}1548$', r'${\rm HI}1215-{\rm OVI}1031$',
                       r'${\rm CII}1334-{\rm CIV}154$', r'${\rm CII}1334-{\rm OVI}1031$', r'${\rm CIV}1548-{\rm OVI}1031$']
    
    x = [0.505, 0.48, 0.48, 0.49, 0.465, 0.445]
    redshift = 0.
    quench = quench_thresh(redshift)
    dv = np.arange(5, 105, 5)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 3
    logN_min = 12.

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')
    ax = ax.flatten()

    for l, line in enumerate(line_b):

        line_b_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line_b[l]}.h5'

        for i in range(len(fr200)):

            with h5py.File(line_b_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:] 
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > logN_min) & (all_chisq < chisq_lim)
            all_ids = all_ids[mask]
            Ntotal = len(all_ids)
            faligned = np.zeros(len(dv))
            
            align_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a[l]}_{line_b[l]}_chisq{chisq_lim}.h5'

            with h5py.File(align_file, 'r') as hf:
                all_dv = hf[f'dv_{fr200[i]}'][:]
                all_N_a = hf[f'{line_a[l]}_log_N_{fr200[i]}'][:]
                all_N_b = hf[f'{line_b[l]}_log_N_{fr200[i]}'][:]
                
            mask = (all_N_a > logN_min) & (all_N_b > logN_min) 
            
            for j in range(len(dv)):
                dv_mask = all_dv < dv[j]
                faligned[j] = len(all_N_b[mask* dv_mask]) / Ntotal

            ax[l].plot(dv, faligned, label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]), c=colors[i], lw=1)

        ax[l].set_xlim(0, dv[-1])
        ax[l].set_ylim(0., 1.0)

        ax[l].annotate(plot_line_pairs[l], xy=(x[l], 0.89), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$|\Delta v\ ({\rm km s}^{-1}) | $')
     
        if l in [0, 3]:
            ax[l].set_ylabel(r'$ f_{\rm aligned}$')
        
    ax[0].legend(loc=4, fontsize=12)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_faligned_chisq{chisq_lim}.png')
    plt.show()
    plt.close()


