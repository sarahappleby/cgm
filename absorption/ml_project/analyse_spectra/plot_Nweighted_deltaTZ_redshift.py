import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]

    snaps = ['105', '125', '137', '151']
    redshift_labels = [r'$z = 1$', r'$z = 0.5$', r'$z = 0.25$', r'$z = 0$']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    line_ev = np.log10([13.6, 15.04, 24.38, 33.49, 64.49, 138.1]) # in eV
    adjust_x = [0.015, 0.025, 0.02, 0.025, 0.02, 0.02]
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}

    N_min = 12.
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]
    deltath = 2.046913
    Tth = 5.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
   
    idelta = 1. / (len(snaps) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.25, .9)
    redshift_colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    fig, ax = plt.subplots(3, 1, figsize=(7, 6.5), sharey='row', sharex='col')
    ax = ax.flatten()

    for i, snap in enumerate(snaps):

        chisq_lim = chisq_lim_dict[f'snap_{snap}']

        snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
        s = pg.Snapshot(snapfile)
        redshift = s.redshift
        rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
        cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

        for l, line in enumerate(lines):

            results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

            all_Z = []
            all_T = []
            all_D = []
            all_N = []
            all_ew = []
            all_chisq = []

            for j in range(len(fr200)):

                with h5py.File(results_file, 'r') as hf:
                    all_Z.extend(hf[f'log_Z_{fr200[j]}r200'][:] - np.log10(zsolar[l]))
                    all_T.extend(hf[f'log_T_{fr200[j]}r200'][:])
                    all_D.extend(hf[f'log_rho_{fr200[j]}r200'][:] - np.log10(cosmic_rho))
                    all_N.extend(hf[f'log_N_{fr200[j]}r200'][:])
                    all_ew.extend(hf[f'ew_{fr200[j]}r200'][:]) 
                    all_chisq.extend(hf[f'chisq_{fr200[j]}r200'][:])

            all_Z = np.array(all_Z)
            all_T = np.array(all_T)
            all_D = np.array(all_D)
            all_N = np.array(all_N)
            all_ew = np.array(all_ew)
            all_chisq = np.array(all_chisq)

            mask = (all_N > N_min) * (all_chisq < chisq_lim[l]) * (all_ew >= 0.)
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_D = all_D[mask]
            all_N = all_N[mask]

            order = np.argsort(all_D)
            weighted_D = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_D_25 = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_D_75 = all_D[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
            order = np.argsort(all_T)
            weighted_T = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_T_25 = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_T_75 = all_T[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
            order = np.argsort(all_Z)
            weighted_Z = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.5))]
            weighted_Z_25 = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.25))]
            weighted_Z_75 = all_Z[order][np.argmin(np.abs(np.nancumsum(all_N[order]) / np.nansum(all_N) - 0.75))]
    
            if i == 0:
                ax[0].errorbar(line_ev[l], weighted_D, color=redshift_colors[i], yerr=np.array([[weighted_D - weighted_D_25, weighted_D_75 - weighted_D,]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)
                ax[1].errorbar(line_ev[l], weighted_T, color=redshift_colors[i], yerr=np.array([[weighted_T - weighted_T_25, weighted_T_75 - weighted_T,]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)
                ax[2].errorbar(line_ev[l], weighted_Z, color=redshift_colors[i], yerr=np.array([[weighted_Z - weighted_Z_25, weighted_Z_75 - weighted_Z,]]).T,
                                  lw=1, ls='None', marker='None', capsize=2)

            ax[0].scatter(line_ev[l], weighted_D, color=redshift_colors[i])
            ax[1].scatter(line_ev[l], weighted_T, color=redshift_colors[i])
            if l == 0:
                ax[2].scatter(line_ev[l], weighted_Z, color=redshift_colors[i], label=redshift_labels[i])
            else:
                ax[2].scatter(line_ev[l], weighted_Z, color=redshift_colors[i])

            if i == len(snaps) -1:
                if line == 'H1215':
                    ax[0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], 0.75), fontsize=13)
                else:
                    ax[0].annotate(plot_lines[l], xy=(line_ev[l] - adjust_x[l], np.min(weighted_D - 0.4)), fontsize=13)

    ax[0].axhline(deltath, ls=':', c='k', lw=1)
    ax[1].axhline(Tth, ls=':', c='k', lw=1)

    ax[2].legend(loc=4, fontsize=12)
    
    ax[0].set_ylim(0.7, 4.25)
    ax[1].set_ylim(4, 5.7)
    ax[2].set_ylim(-1.5, )

    ax[2].set_xlabel(r'${\rm log }(E / {\rm eV})$')
    ax[0].set_ylabel(r'${\rm log }\delta$')
    ax[1].set_ylabel(r'${\rm log } (T / {\rm K})$')
    ax[2].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_redshift_Nweighted_deltaTZ.png')
    plt.clf()
