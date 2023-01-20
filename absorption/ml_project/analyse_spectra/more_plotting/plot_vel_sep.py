import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

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

    if snap == '151':
        redshift = 0.

    quench = quench_thresh(redshift)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    redshift = 0.
    chisq_lim = 2.5
    N_min = 12.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    delta_dv = 50.
    min_dv = 0.
    nbins_dv = 12
    dv_bins = np.arange(min_dv, min_dv+(nbins_dv+1)*delta_dv, delta_dv)
    plot_dv = get_bin_middle(dv_bins)

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    sf_color = [cmap(i) for i in icolor]
    gv_color = [cmap(i) for i in icolor]
    q_color = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    fig, ax = plt.subplots(len(lines), 3, figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_pos_dv = np.abs(hf[f'pos_dv_{fr200[i]}r200'][:])
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim)
            all_N = all_N[mask]
            all_pos_dv = all_pos_dv[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten() 
            all_ssfr = ssfr[idx]
           
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)   

            N_sf_25 = np.zeros(nbins_dv)
            N_sf_med = np.zeros(nbins_dv)
            N_sf_75 = np.zeros(nbins_dv)
            N_gv_25 = np.zeros(nbins_dv)
            N_gv_med = np.zeros(nbins_dv)
            N_gv_75 = np.zeros(nbins_dv)
            N_q_25 = np.zeros(nbins_dv)
            N_q_med = np.zeros(nbins_dv)
            N_q_75 = np.zeros(nbins_dv)

            for j in range(nbins_dv):

                dv_mask = (all_pos_dv > dv_bins[j]) & (all_pos_dv < dv_bins[j+1])

                if len(all_pos_dv[dv_mask]) > 0:

                    N_sf_25[j] = np.nanpercentile(all_N[dv_mask*sf_mask], 25)
                    N_sf_med[j] = np.nanpercentile(all_N[dv_mask*sf_mask], 50)
                    N_sf_75[j] = np.nanpercentile(all_N[dv_mask*sf_mask], 75)
                    
                    N_gv_25[j] = np.nanpercentile(all_N[dv_mask*gv_mask], 25)
                    N_gv_med[j] = np.nanpercentile(all_N[dv_mask*gv_mask], 50)
                    N_gv_75[j] = np.nanpercentile(all_N[dv_mask*gv_mask], 75)
                    
                    N_q_25[j] = np.nanpercentile(all_N[dv_mask*q_mask], 25)
                    N_q_med[j] = np.nanpercentile(all_N[dv_mask*q_mask], 50)
                    N_q_75[j] = np.nanpercentile(all_N[dv_mask*q_mask], 75)
            
            ax[l][0].plot(plot_dv, N_sf_med, color=sf_color[i], ls='-', lw=1, marker='None', label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[l][1].plot(plot_dv, N_gv_med, color=gv_color[i], ls='-', lw=1, marker='None', label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[l][2].plot(plot_dv, N_q_med, color=q_color[i], ls='-', lw=1, marker='None', label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))

         
        for j in range(3):
            ax[l][j].set_xlim(0, 600) 
            ax[l][j].set_ylim(12, 16)
            
        if l == 0:   
            ax[l][0].legend()
            ax[l][0].set_title('Star forming')
            ax[l][1].set_title('Green valley')
            ax[l][2].set_title('Quenched')

        if l == len(lines)-1:
            for j in range(3):
                ax[l][j].set_xlabel(r'$\delta v ({\rm km s}^{-1})$')
        if i == 0:
            ax[l][i].set_ylabel(r'${\rm log }(N / {\rm cm}^{-2})$')
            ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_vel_sep.png')
    plt.show()
    plt.clf()


