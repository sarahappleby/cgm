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
    all_dv = np.arange(5., 105., 5.)

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

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        ssfr = sf['ssfr'][:]

    line_a_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line_a}.h5'

    fig, ax = plt.subplots(len(line_b), 4, figsize=(14, 10), sharey='row', sharex='col')
    #fig, ax = plt.subplots(len(line_b), 4, figsize=(14, 6), sharey='row', sharex='col')


    for l, line in enumerate(line_b):

        line_b_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line_b[l]}.h5'

        for i in range(len(fr200)):

            with h5py.File(line_b_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > logN_min)
            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
            all_ssfr = ssfr[idx]
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            Ntotal_all = len(all_ids)
            Ntotal_sf = len(all_ids[sf_mask])
            Ntotal_gv = len(all_ids[gv_mask])
            Ntotal_q = len(all_ids[q_mask])

            faligned_all = np.zeros(len(all_dv))
            faligned_sf = np.zeros(len(all_dv))
            faligned_gv = np.zeros(len(all_dv))
            faligned_q = np.zeros(len(all_dv))
            
            align_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b[l]}_{fr200[i]}r200.h5'

            for j in range(len(all_dv)):
                with h5py.File(align_file, 'r') as hf:
                    all_N_a = hf[f'{line_a}_log_N_{all_dv[j]}kms'][:]
                    all_N_b = hf[f'{line_b[l]}_log_N_{all_dv[j]}kms'][:]
                    all_ids = hf[f'ids_{all_dv[j]}kms'][:]
                mask = (all_N_a > logN_min) & (all_N_b > logN_min) 
                
                all_ids = all_ids[mask]
                idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
                all_ssfr = ssfr[idx]
                sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)
                
                faligned_all[j] = len(all_ids) / Ntotal_all
                faligned_sf[j] = len(all_ids[sf_mask]) / Ntotal_sf
                faligned_gv[j] = len(all_ids[gv_mask]) / Ntotal_gv
                faligned_q[j] = len(all_ids[q_mask]) / Ntotal_q

            ax[l][0].plot(all_dv, faligned_all, label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]), c=colors[i], lw=1)
            ax[l][1].plot(all_dv, faligned_sf, c=colors[i], lw=1)
            ax[l][2].plot(all_dv, faligned_gv, c=colors[i], lw=1)
            ax[l][3].plot(all_dv, faligned_q, c=colors[i], lw=1)

        for i in range(4):
            ax[l][i].set_xlim(0, all_dv[-1])
            ax[l][i].set_ylim(0., 1.0)

        if l == 0:
            ax[l][0].legend(loc=2)
            ax[l][0].set_title('All')
            ax[l][1].set_title('Star forming')
            ax[l][2].set_title('Green valley')
            ax[l][3].set_title('Quenched')

        if l == len(line_b)-1:
            for i in range(4):
                ax[l][i].set_xlabel(r'$|v({\rm HI}) - v({\rm Ion}) ({\rm km s}^{-1}) | $')
                # ax[l][i].set_xlabel(r'$|v({\rm CII}) - v({\rm Ion}) ({\rm km s}^{-1}) | $')
        ax[l][0].set_ylabel(r'$ f_{\rm aligned}$')
        ax[l][0].annotate(plot_line_b[l], xy=(0.7, 0.85), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_aligned_{line_a}_ssfr_split.png')
    plt.show()
    plt.close()


