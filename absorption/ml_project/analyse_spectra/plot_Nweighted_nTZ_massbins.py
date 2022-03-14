import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    lines = ["H1215", "CII1334", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm CII}1334$', r'${\rm OVI}1031$']

    ion_mass = np.array([pg.UnitArr(pg.analysis.absorption_spectra.lines[line]['atomwt']) * pg.physics.m_u for line in lines])
    chisq_lim = 2.5
    N_min = 12.
    zsolar = 0.0134

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
   
    delta_m = 0.5
    min_m = 11.5
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    plot_mass = get_bin_middle(mass_bins)

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mhalo = sf['halo_mass'][:]
   
    fig, ax = plt.subplots(3, len(lines), figsize=(10, 6.5), sharey='row', sharex='col')

    for l, line in enumerate(lines):
        
        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            weighted_n = np.zeros(nbins_m)
            weighted_n_25 = np.zeros(nbins_m)
            weighted_n_75 = np.zeros(nbins_m)
            weighted_T = np.zeros(nbins_m)
            weighted_T_25 = np.zeros(nbins_m)
            weighted_T_75 = np.zeros(nbins_m)
            weighted_Z = np.zeros(nbins_m)
            weighted_Z_25 = np.zeros(nbins_m)
            weighted_Z_75 = np.zeros(nbins_m)

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar)
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_n = hf[f'log_rho_{fr200[i]}r200'][:] - np.log10(ion_mass[l])
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim)
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_n = all_n[mask]
            all_N = all_N[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
            all_mhalo = mhalo[idx]

            for j in range(nbins_m):

                mass_mask = (all_mhalo > mass_bins[j]) & (all_mhalo < mass_bins[j+1])
                if len(all_mhalo[mass_mask]) > 0:

                    order = np.argsort(all_n[mass_mask])
                    weighted_n[j] = all_n[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                    weighted_n_25[j] = all_n[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                    weighted_n_75[j] = all_n[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]

                    order = np.argsort(all_T[mass_mask])
                    weighted_T[j] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                    weighted_T_25[j] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                    weighted_T_75[j] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]

                    order = np.argsort(all_Z[mass_mask])
                    weighted_Z[j] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                    weighted_Z_25[j] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                    weighted_Z_75[j] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]


            if i == 0:
                ax[0][l].errorbar(plot_mass, weighted_n, xerr=0.5*delta_m, color=colors[i], yerr=[weighted_n - weighted_n_25, weighted_n_75 - weighted_n,], 
                                  lw=1, ls='None', marker='None', capsize=2)

            ax[0][l].scatter(plot_mass, weighted_n, color=colors[i], label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[1][l].scatter(plot_mass, weighted_T, color=colors[i])
            ax[2][l].scatter(plot_mass, weighted_Z, color=colors[i])
    
        if l == len(lines)-1:
            ax[0][l].legend(loc=0, fontsize=12)
        #ax[0].set_ylim(-3.75, -0.5)
        for i in range(3):
            ax[i][l].set_xlim(min_m-0.1, min_m+(nbins_m)*delta_m+0.1)

        ax[2][l].set_xlabel(r'$\log\ (M_{\rm halo} / M_{\odot})$')
        if l == 0:
            ax[0][l].set_ylabel(r'${\rm log }(n / {\rm cm}^{-3})$')
            ax[1][l].set_ylabel(r'${\rm log } (T / {\rm K})$')
            ax[2][l].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')
        ax[0][l].set_title(plot_lines[l])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Nweighted_nTZ_mhalo.png')
    plt.show()
    plt.clf()
