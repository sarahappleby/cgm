import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13.5)

def plot_nweighted(results_file, line, inner_outer, line_ev, zsolar, cosmic_rho, N_min, chisq_lim):

    weighted = {}

    weighted['D'] = np.zeros(2)
    weighted['D_25'] = np.zeros(2)
    weighted['D_75'] = np.zeros(2)
    weighted['T'] = np.zeros(2)
    weighted['T_25'] = np.zeros(2)
    weighted['T_75'] = np.zeros(2)
    weighted['Z'] = np.zeros(2)
    weighted['Z_25'] = np.zeros(2)
    weighted['Z_75'] = np.zeros(2)

    for i in range(len(inner_outer)):

        all_Z = []
        all_T = []
        all_D = []
        all_ids = []
        all_chisq = []
        all_N = []

        for j in range(len(inner_outer[i])):

            with h5py.File(results_file, 'r') as hf:
                all_Z.extend(hf[f'log_Z_{inner_outer[i][j]}r200'][:] - np.log10(zsolar))
                all_T.extend(hf[f'log_T_{inner_outer[i][j]}r200'][:])
                all_D.extend(hf[f'log_rho_{inner_outer[i][j]}r200'][:] - np.log10(cosmic_rho))
                all_N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                all_chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                all_ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])

        all_Z = np.array(all_Z)
        all_T = np.array(all_T)
        all_D = np.array(all_D)
        all_ids = np.array(all_ids)
        all_chisq = np.array(all_chisq)
        all_N = np.array(all_N)

        mask = (all_N > N_min) * (all_chisq < chisq_lim)
        all_Z = all_Z[mask]
        all_T = all_T[mask]
        all_D = all_D[mask]
        all_ids = all_ids[mask]
        all_N = all_N[mask]

        if len(all_N) == 0:
            continue

        order = np.argsort(all_D)
        weighted['D'][i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
        weighted['D_25'][i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
        weighted['D_75'][i] = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
        order = np.argsort(all_T)
        weighted['T'][i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
        weighted['T_25'][i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
        weighted['T_75'][i] = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
        order = np.argsort(all_Z)
        weighted['Z'][i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
        weighted['Z_25'][i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
        weighted['Z_75'][i] = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]

        """
        if i == 0:
            ax[0].errorbar(line_ev, weighted_D, color=colors[i], yerr=np.array([[weighted_D - weighted_D_25, weighted_D_75 - weighted_D,]]).T,
                              lw=1, ls='None', marker='None', capsize=2)
            ax[1].errorbar(line_ev, weighted_T, color=colors[i], yerr=np.array([[weighted_T - weighted_T_25, weighted_T_75 - weighted_T,]]).T,
                              lw=1, ls='None', marker='None', capsize=2)
            ax[2].errorbar(line_ev, weighted_Z, color=colors[i], yerr=np.array([[weighted_Z - weighted_Z_25, weighted_Z_75 - weighted_Z,]]).T,
                              lw=1, ls='None', marker='None', capsize=2)

        ax[0].scatter(line_ev, weighted_D, color=colors[i], marker=marker)
        ax[1].scatter(line_ev, weighted_T, color=colors[i], marker=marker)
        ax[2].scatter(line_ev, weighted_Z, color=colors[i], marker=marker)
        """

    return weighted 

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = ['MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    line_ev = np.log10([15.04, 24.38, 33.49, 64.49, 138.1]) # in eV
    adjust_x = [0.025, 0.02, 0.025, 0.02, 0.02]
    y = [0.06, 0.3, 0.64, 0.13, 0.24]
    chisq_lim = [4.5, 20., 20., 20., 7.1, 2.8]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    zsolar = [7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]
    deltath = 2.046913
    Tth = 5.

    inner_outer = [[0.25, 0.5], [0.75, 1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    
    idelta = 0.6
    icolor = np.arange(0.15, 0.75+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    phys = ['D', 'T', 'Z']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]
    
    fig, ax = plt.subplots(3, 1, figsize=(7, 6.5), sharey='row', sharex='col')
    ax = ax.flatten()
   
    rho_lines = []
    for i in range(len(colors)):
        rho_lines.append(Line2D([0,1],[0,1], color=colors[i], ls='-', lw=1, alpha=0.6))
    leg = ax[0].legend(rho_lines, rho_labels, loc=1, fontsize=12)
    ax[0].add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/{model}_{wind}_{snap}_no_uvb_fit_lines_{line}.h5'
        weighted_uvb = plot_nweighted(results_file, line, inner_outer, line_ev[l], zsolar[l], cosmic_rho, N_min[l], chisq_lim[l])
        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
        weighted_norm = plot_nweighted(results_file, line, inner_outer, line_ev[l], zsolar[l], cosmic_rho, N_min[l], chisq_lim[l])

        for i in range(len(phys)):

            diff = weighted_uvb[phys[i]] - weighted_norm[phys[i]]
            
            err_uvb = np.array([weighted_uvb[f'{phys[i]}'] - weighted_uvb[f'{phys[i]}_25'], weighted_uvb[f'{phys[i]}_75'] - weighted_uvb[f'{phys[i]}']])
            err_norm = np.array([weighted_norm[f'{phys[i]}'] - weighted_norm[f'{phys[i]}_25'], weighted_norm[f'{phys[i]}_75'] - weighted_norm[f'{phys[i]}']])
       
            err = np.sqrt(err_uvb**2 + err_norm**2)
            err = err.T

            for j in range(2):
                ax[i].errorbar(line_ev[l], diff[j], color=colors[j], yerr=np.reshape(err[j], (2, 1)),
                                lw=1, ls='None', marker='o', capsize=2, alpha=0.6)
            
            if i == 0:
                bottom = diff - err[:, 0]
                ax[0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], y[l]), fontsize=13)
    
    ax[0].axhline(0, ls=':', c='k', lw=1)
    ax[1].axhline(0, ls=':', c='k', lw=1)
    ax[2].axhline(0, ls=':', c='k', lw=1)

    ax[2].set_xlabel(r'${\rm log }(E / {\rm eV})$')
    ax[0].set_ylabel(r'${\rm log }(\delta_{\rm No UVB} / \delta_{\rm UVB})$')
    ax[1].set_ylabel(r'${\rm log } (T_{\rm No UVB} / T_{\rm UVB})$')
    ax[2].set_ylabel(r'${\rm log} (Z_{\rm No UVB} / Z_{\rm UVB})$')

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_no_uvb_diff_Nweighted_deltaTZ.pdf', format='pdf')
    plt.show()
    plt.close()
