import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    line_ev = np.log10([13.6, 15.04, 24.38, 33.49, 64.49, 138.1]) # in eV
    adjust_x = [0.015, 0.025, 0.02, 0.025, 0.02, 0.02]
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]
    deltath = 2.046913
    Tth = 5.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
    
    fig, ax = plt.subplots(3, 3, figsize=(10, 6.5), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        weighted_D = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_D_25 = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_D_75 = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_T = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_T_25 = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_T_75 = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_Z = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_Z_25 = np.zeros((len(mass_plot_titles), len(fr200)))
        weighted_Z_75 = np.zeros((len(mass_plot_titles), len(fr200)))

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_D = hf[f'log_rho_{fr200[i]}r200'][:] - np.log10(cosmic_rho)
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min[l]) * (all_chisq < chisq_lim[l])
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_D = all_D[mask]
            all_ids = all_ids[mask]
            all_N = all_N[mask]
            
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
            all_mass = mass[idx]

            for j in range(len(mass_plot_titles)):

                mass_mask = (all_mass > mass_bins[j]) & (all_mass < mass_bins[j+1])

                order = np.argsort(all_D[mass_mask])
                weighted_D[j][i] = all_D[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                weighted_D_25[j][i] = all_D[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                weighted_D_75[j][i] = all_D[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]
                order = np.argsort(all_T[mass_mask])
                weighted_T[j][i] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                weighted_T_25[j][i] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                weighted_T_75[j][i] = all_T[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]
                order = np.argsort(all_Z[mass_mask])
                weighted_Z[j][i] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.5))]
                weighted_Z_25[j][i] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.25))]
                weighted_Z_75[j][i] = all_Z[mass_mask][order][np.argmin(np.abs(np.nancumsum(all_N[mass_mask][order]) / np.nansum(all_N[mass_mask]) - 0.75))]
    
                if i == 0:
                    ax[0][j].errorbar(line_ev[l], weighted_D[j][i], color=colors[i], yerr=np.array([[weighted_D[j][i] - weighted_D_25[j][i], weighted_D_75[j][i] - weighted_D[j][i],]]).T,
                                   lw=1, ls='None', marker='None', capsize=2)
                    ax[1][j].errorbar(line_ev[l], weighted_T[j][i], color=colors[i], yerr=np.array([[weighted_T[j][i] - weighted_T_25[j][i], weighted_T_75[j][i] - weighted_T[j][i],]]).T,
                                   lw=1, ls='None', marker='None', capsize=2)
                    ax[2][j].errorbar(line_ev[l], weighted_Z[j][i], color=colors[i], yerr=np.array([[weighted_Z[j][i] - weighted_Z_25[j][i], weighted_Z_75[j][i] - weighted_Z[j][i],]]).T,
                                   lw=1, ls='None', marker='None', capsize=2)


                ax[0][j].scatter(line_ev[l], weighted_D[j][i], color=colors[i])
                ax[1][j].scatter(line_ev[l], weighted_T[j][i], color=colors[i])
                if l == 0:
                    ax[2][j].scatter(line_ev[l], weighted_Z[j][i], color=colors[i], label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
                else:
                    ax[2][j].scatter(line_ev[l], weighted_Z[j][i], color=colors[i])

        ax[0][0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], np.min(weighted_D - 0.35)), fontsize=11)

    ax[2][2].legend(loc=4, fontsize=12)

    for j in range(3):
        ax[0][j].axhline(deltath, ls=':', c='k', lw=1)
        ax[1][j].axhline(Tth, ls=':', c='k', lw=1)

        ax[0][j].set_ylim(1, 4.)
        ax[1][j].set_ylim(4, 5.7)
        ax[2][j].set_ylim(-1.75, )

        ax[2][j].set_xlabel(r'${\rm log }(E / {\rm eV})$')

        ax[0][j].set_title(mass_plot_titles[j])

    ax[0][0].set_ylabel(r'${\rm log }\delta$')
    ax[1][0].set_ylabel(r'${\rm log } (T / {\rm K})$')
    ax[2][0].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')

    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    #ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_Nweighted_deltaTZ_mass.pdf', format='pdf')
    plt.clf()
