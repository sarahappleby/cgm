import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import caesar
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

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

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)

    chisq_lim = 2.5
    N_min = 12.

    dv_max = 600.
    dv_min = 0.
    deltadv = 50.
    dv_bins = np.arange(dv_min, dv_max+deltadv, deltadv)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        ssfr = sf['ssfr'][:]

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_pos_dv = np.abs(hf[f'pos_dv_{fr200[i]}r200'][:])
                #all_ew = hf[f'ew_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim)
            all_N = all_N[mask]
            all_pos_dv = all_pos_dv[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten() 
            all_ssfr = ssfr[idx]

            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            ax[l][i].hist(all_pos_dv, bins=dv_bins, density=True, color='k', histtype='step', label='All')
            ax[l][i].hist(all_pos_dv[sf_mask], bins=dv_bins, density=True, color=cb_blue, histtype='step', label='SF')
            ax[l][i].hist(all_pos_dv[gv_mask], bins=dv_bins, density=True, color=cb_green, histtype='step', label='GV')
            ax[l][i].hist(all_pos_dv[q_mask], bins=dv_bins, density=True, color=cb_red, histtype='step', label='Q')

            ax[l][i].set_xlim(0, 600) 
            #ax[l][i].set_ylim(0, 3.)
            
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'$\delta v ({\rm kms s}^{-1})$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm Frequency}$')
                ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')
                if l == 0:
                    ax[l][i].legend(loc=3)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_vel_sep_hist.png')
    plt.show()
    plt.clf()


