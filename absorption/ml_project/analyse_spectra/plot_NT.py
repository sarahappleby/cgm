import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import cm
import numpy as np
import h5py
import pygad as pg
import sys

sys.path.append('/disk04/sapple/tools')
import plotmedian as pm

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


def quench_thresh(z): # in units of yr^-1
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.1, 1.0)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']

    x = [0.74, 0.68, 0.715, 0.69, 0.695, 0.7]
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    Tth = 5

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    nmin = 15
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    snapfile = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        all_ssfr = sf['ssfr'][:]

    cmap = cm.get_cmap('viridis')
    cmap = truncate_colormap(cmap, 0.1, 0.9)
    norm = colors.BoundaryNorm(np.arange(0.125, 1.625, 0.25), cmap.N)

    fig, ax = plt.subplots(len(lines), 3, figsize=(14, 13), sharey='row', sharex='col')

    rho_lines = []
    rho_lines.append(Line2D([0,1],[0,1], color='plum', ls='--', lw=2))
    rho_lines.append(Line2D([0,1],[0,1], color='plum', ls='-', lw=2))
    leg = ax[0][0].legend(rho_lines, rho_labels, loc=4, fontsize=12)
    ax[0][0].add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        N = []
        T = []
        chisq = []
        ids = []
        all_r = []

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                T.extend(hf[f'log_T_{fr200[i]}r200'][:])
                chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                ids.extend(hf[f'ids_{fr200[i]}r200'][:])
                all_r.extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

        N = np.array(N)
        T = np.array(T)
        chisq = np.array(chisq)
        ids = np.array(ids)
        all_r = np.array(all_r)

        mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
        T = T[mask]
        N = N[mask]
        all_r = all_r[mask]
        ids = ids[mask]

        idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
        ssfr = all_ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr)

        inner_mask = all_r < 1.0
        silly_mask = N < 18.

        plot_order = np.arange(len(N[sf_mask]))
        np.random.shuffle(plot_order)
        im = ax[l][0].scatter(N[sf_mask][plot_order], T[sf_mask][plot_order], c=all_r[sf_mask][plot_order], cmap=cmap, norm=norm, s=1.5)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[sf_mask*inner_mask*silly_mask], T[sf_mask*inner_mask*silly_mask], stat='median')
        ax[l][0].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='--')
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[sf_mask*~inner_mask*silly_mask], T[sf_mask*~inner_mask*silly_mask], stat='median')
        ax[l][0].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='-')

        plot_order = np.arange(len(N[gv_mask]))
        np.random.shuffle(plot_order)
        im = ax[l][1].scatter(N[gv_mask][plot_order], T[gv_mask][plot_order], c=all_r[gv_mask][plot_order], cmap=cmap, norm=norm, s=1.5)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[gv_mask*inner_mask*silly_mask], T[gv_mask*inner_mask*silly_mask], stat='median')
        ax[l][1].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='--')
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[gv_mask*~inner_mask*silly_mask], T[gv_mask*~inner_mask*silly_mask], stat='median')
        ax[l][1].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='-')

        plot_order = np.arange(len(N[q_mask]))
        np.random.shuffle(plot_order)
        im = ax[l][2].scatter(N[q_mask][plot_order], T[q_mask][plot_order], c=all_r[q_mask][plot_order], cmap=cmap, norm=norm, s=1.5)
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[q_mask*inner_mask*silly_mask], T[q_mask*inner_mask*silly_mask], stat='median')
        ax[l][2].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='--')
        bin_cent,ymean,ysiglo,ysighi,ndata = pm.runningmedian(N[q_mask*~inner_mask*silly_mask], T[q_mask*~inner_mask*silly_mask], stat='median')
        ax[l][2].plot(bin_cent[ndata>nmin], ymean[ndata>nmin], c='plum', lw=2, ls='-')

        if l == 0:
            ax[l][0].set_title('Star forming')
            ax[l][1].set_title('Green valley')
            ax[l][2].set_title('Quenched')

        if l == len(lines)-1:
            for i in range(3):
                ax[l][i].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        
        ax[l][0].set_ylabel(r'${\rm log } (T / {\rm K})$')
        
        ax[l][2].annotate(plot_lines[lines.index(line)], xy=(x[l], 0.1), xycoords='axes fraction',
                          fontsize=14, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        for i in range(3):
            ax[l][i].set_xlim(np.min(N_min), 18)
            ax[l][i].set_ylim(3, 6)
            ax[l][i].axhline(Tth, ls=':', c='k', lw=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.24, 0.02, 0.51])
    fig.colorbar(im, cax=cbar_ax, ticks=fr200, label=r'$\rho / r_{200}$')
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_NT_r200.pdf', format='pdf')
    plt.close()
