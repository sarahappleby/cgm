import matplotlib.pyplot as plt
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

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

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

    vel_range = 600.
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    
    redshift = 0.
    quench = quench_thresh(redshift)
    chisq_lim = 2.5

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    N_min = 12.
    N_max = 18.
    delta_N = 0.5
    bins_N = np.arange(N_min, N_max+delta_N, delta_N)
    plot_N = get_bin_middle(bins_N)

    path_length_file = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/path_lengths.h5'
    if not os.path.isfile(path_length_file):
        create_path_length_file(vel_range, lines, redshift, path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        fig, ax = plt.subplots(1, len(fr200), figsize=(14, 5), sharey='row', sharex='col')
        ax = ax.flatten()

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_b = hf[f'b_{fr200[i]}r200'][:]
                all_l = hf[f'l_{fr200[i]}r200'][:]
                #all_ew = hf[f'ew_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim)
            all_N = all_N[mask]
            all_b = all_b[mask]
            all_l = all_l[mask]
            #all_ew = all_ew[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten() 
            all_mass = mass[idx]
            all_ssfr = ssfr[idx]

            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            cddf_all = np.zeros(len(plot_N))
            cddf_sf = np.zeros(len(plot_N))
            cddf_gv = np.zeros(len(plot_N))
            cddf_q = np.zeros(len(plot_N))

            dX = compute_dX(model, wind, snap, lines, len(all_ids), path_lengths)    
     
            for j in range(len(plot_N)):
                N_mask = (all_N > N_min + j*delta_N) & (all_N < N_min + (j+1)*delta_N)
                cddf_all[j] = len(all_N[N_mask])
                cddf_sf[j] = len(all_N[N_mask*sf_mask])
                cddf_gv[j] = len(all_N[N_mask*gv_mask])
                cddf_q[j] = len(all_N[N_mask*q_mask])

            cddf_all /= (delta_N * dX[0])
            cddf_sf /= (delta_N * dX[0])
            cddf_gv /= (delta_N * dX[0])
            cddf_q /= (delta_N * dX[0])

            ax[i].plot(plot_N, np.log10(cddf_all + 1e-1), label='All', c='k', lw=1)
            ax[i].plot(plot_N, np.log10(cddf_sf + 1e-1), label='SF', c=cb_blue, lw=1)
            ax[i].plot(plot_N, np.log10(cddf_gv + 1e-1), label='GV', c=cb_green, lw=1)
            ax[i].plot(plot_N, np.log10(cddf_q + 1e-1), label='Q', c=cb_red, lw=1)
   
            ax[i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[i].set_xlim(12, 18)
            ax[i].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
            if i == 0:
                ax[i].set_ylabel(r'${\rm log }( \delta^2 N / \delta X \delta N )$')
                ax[i].legend()

        plt.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_{line}.png')
        plt.clf()


