import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = ['MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    line_ev = np.log10([15.04, 24.38, 33.49, 64.49, 138.1]) # in eV
    adjust_x = [0.025, 0.02, 0.025, 0.02, 0.02]
    chisq_lim = [4.5, 20., 20., 20., 7.1, 2.8]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    N_min = 12.
    zsolar = [7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]
    deltath = 2.046913
    Tth = 5.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    
    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]
    
    fig, ax = plt.subplots(3, 1, figsize=(7, 6.5), sharey='row', sharex='col')
    ax = ax.flatten()

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/{model}_{wind}_{snap}_no_uvb_fit_lines_{line}.h5'

        weighted_D = np.zeros(len(fr200))
        weighted_D_25 = np.zeros(len(fr200))
        weighted_D_75 = np.zeros(len(fr200))
        weighted_T = np.zeros(len(fr200))
        weighted_T_25 = np.zeros(len(fr200))
        weighted_T_75 = np.zeros(len(fr200))
        weighted_Z = np.zeros(len(fr200))
        weighted_Z_25 = np.zeros(len(fr200))
        weighted_Z_75 = np.zeros(len(fr200))

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_D = hf[f'log_rho_{fr200[i]}r200'][:] - np.log10(cosmic_rho) 
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim[l])
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_D = all_D[mask]
            all_ids = all_ids[mask]
            all_N = all_N[mask]

            if len(all_N) == 0:
                continue

            order = np.argsort(all_D)
            weighted_D[i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_D_25[i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_D_75[i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
            order = np.argsort(all_T)
            weighted_T[i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_T_25[i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_T_75[i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
            order = np.argsort(all_Z)
            weighted_Z[i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_Z_25[i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_Z_75[i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
    
            if i == 0:
                ax[0].errorbar(line_ev[l], weighted_D[i], color=colors[i], yerr=np.array([[weighted_D[i] - weighted_D_25[i], weighted_D_75[i] - weighted_D[i],]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)
                ax[1].errorbar(line_ev[l], weighted_T[i], color=colors[i], yerr=np.array([[weighted_T[i] - weighted_T_25[i], weighted_T_75[i] - weighted_T[i],]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)
                ax[2].errorbar(line_ev[l], weighted_Z[i], color=colors[i], yerr=np.array([[weighted_Z[i] - weighted_Z_25[i], weighted_Z_75[i] - weighted_Z[i],]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)


            ax[0].scatter(line_ev[l], weighted_D[i], color=colors[i])
            ax[1].scatter(line_ev[l], weighted_T[i], color=colors[i])
            if l == 0:
                ax[2].scatter(line_ev[l], weighted_Z[i], color=colors[i], label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            else:
                ax[2].scatter(line_ev[l], weighted_Z[i], color=colors[i])

        ax[0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], np.min(weighted_D[weighted_D != 0.] - 0.45)))

    ax[0].axhline(deltath, ls=':', c='k', lw=1)
    ax[1].axhline(Tth, ls=':', c='k', lw=1)

    ax[2].legend(loc=4, fontsize=12)
    
    ax[0].set_ylim(1.5, 5.)
    ax[1].set_ylim(3, 5.5)
    ax[2].set_ylim(-1.5, )

    ax[2].set_xlabel(r'${\rm log }(E / {\rm eV})$')
    ax[0].set_ylabel(r'${\rm log }\delta$')
    ax[1].set_ylabel(r'${\rm log } (T / {\rm K})$')
    ax[2].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_no_uvb_Nweighted_deltaTZ_chisqion.png')
    plt.clf()
