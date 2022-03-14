import matplotlib.pyplot as plt
from matplotlib import cm
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

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    line_ev = np.log10([13.6, 15.04, 24.38, 33.49, 64.49, 138.1]) # in eV
    adjust_x = [0.015, 0.025, 0.02, 0.025, 0.02, 0.02]

    ion_mass = np.array([pg.UnitArr(pg.analysis.absorption_spectra.lines[line]['atomwt']) * pg.physics.m_u for line in lines])
    chisq_lim = 2.5
    N_min = 12.
    zsolar = 0.0134

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

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        weighted_n = np.zeros(len(fr200))
        weighted_T = np.zeros(len(fr200))
        weighted_Z = np.zeros(len(fr200))

        for i in range(len(fr200)):

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
            all_ids = all_ids[mask]
            all_N = all_N[mask]

            order = np.argsort(all_n)
            weighted_n[i] = all_n[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            order = np.argsort(all_T)
            weighted_T[i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            order = np.argsort(all_Z)
            weighted_Z[i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]

            if l == 0:
                ax[0].scatter(line_ev[l], weighted_n[i], color=colors[i], label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            else:
                ax[0].scatter(line_ev[l], weighted_n[i], color=colors[i])

            ax[1].scatter(line_ev[l], weighted_T[i], color=colors[i])
            ax[2].scatter(line_ev[l], weighted_Z[i], color=colors[i])
    
        ax[0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], np.min(weighted_n - 0.45)))

    ax[0].legend(loc=1, fontsize=12)
    ax[0].set_ylim(-3.75, -0.5)

    ax[2].set_xlabel(r'${\rm log }(E / {\rm eV})$')
    ax[0].set_ylabel(r'${\rm log }(n / {\rm cm}^{-3})$')
    ax[1].set_ylabel(r'${\rm log } (T / {\rm K})$')
    ax[2].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Nweighted_nTZ.png')
    plt.clf()