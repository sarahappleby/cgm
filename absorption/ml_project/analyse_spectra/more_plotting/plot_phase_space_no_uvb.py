import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    cmap = plt.get_cmap('Greys')
    cmap = truncate_colormap(cmap, 0.0, .6)

    lines = ["H1215", "MgII2796", "CII1334"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$']
    cbar_labels = [r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm MgII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$']
    N_min = [12., 11., 12.]
    x = [0.04]*3
    chisq_lim = [4.5, 20., 20.]

    #lines = ["SiIII1206", "CIV1548", "OVI1031"]
    #plot_lines = [r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    #cbar_labels = [r'${\rm log }(N\ {\rm SiIII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CIV} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm OVI} / {\rm cm}^{-2})$']
    #N_min = [11., 12., 12.]
    #x = [0.04]* 3
    #chisq_lim = [20, 7.1, 2.8]

    #width = 0.258
    #height = 0.015
    #vertical_position = 0.95
    #horizontal_position = [0.125, 0.3833, 0.6416]

    width = 0.208
    height = 0.013
    vertical_position = 0.81
    horizontal_position = [0.15, 0.4088, 0.666]
    
    xticks = [[-1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]

    deltath = 2.046913
    Tth = 5.

    snapfile = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    phase_space_file = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}_phase_space.h5'
    with h5py.File(phase_space_file, 'r') as hf:
        rho_overdensity_temp_hist2d = hf['rho_delta_temp'][:]
        rho_overdensity_bins = hf['rho_delta_bins'][:]
        temp_bins = hf['temp_bins'][:]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey='row', sharex='col')
    
    if lines[0] == 'H1215':
        fig.delaxes(ax[0])

    for l, line in enumerate(lines):

        if line == 'H1215': continue 

        results_file = f'/disk04/sapple/data/collisional/results/{model}_{wind}_{snap}_no_uvb_fit_lines_{line}.h5'

        all_T = []
        all_rho = []
        all_N = []
        all_chisq = []
        all_ids = []

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_T.extend(hf[f'log_T_{fr200[i]}r200'][:])
                all_rho.extend(hf[f'log_rho_{fr200[i]}r200'][:])
                all_N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                all_ids.extend(hf[f'ids_{fr200[i]}r200'][:])

        all_T = np.array(all_T)
        all_rho = np.array(all_rho)
        all_N = np.array(all_N)
        all_chisq = np.array(all_chisq)
        all_ids = np.array(all_ids)

        mask = (all_N > N_min[l]) * (all_chisq < chisq_lim[l])
        all_T = all_T[mask]
        all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
        all_ids = all_ids[mask]
        all_N = all_N[mask]

        condensed = (all_T < Tth) & (all_delta_rho > deltath)
        condensed = np.around(len(all_T[condensed])*100 / len(all_T), 1)
        hot_halo = (all_T > Tth) & (all_delta_rho > deltath)
        hot_halo = np.around(len(all_T[hot_halo])*100 / len(all_T), 1)
        whim = (all_T > Tth) & (all_delta_rho < deltath)
        whim = np.around(len(all_T[whim])*100 / len(all_T), 1)
        diffuse = (all_T < Tth) & (all_delta_rho < deltath)
        diffuse = np.around(len(all_T[diffuse])*100 / len(all_T), 1)
        print(f'{line}: {condensed}% Condensed, {hot_halo}% Hot Halo, {whim}% WHIM, {diffuse}% Diffuse')

        ax[l].imshow(np.log10(rho_overdensity_temp_hist2d), extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]), 
                        cmap=cmap)

        ax[l].axhline(Tth, c='k', ls=':', lw=1)
        ax[l].axvline(deltath, c='k', ls=':', lw=1)

        if line == 'H1215':
            im = ax[l].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=16)
        else:
            im = ax[l].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=15)
            
        ax[l].set_xlim(-1, 7)
        ax[l].set_ylim(3, 7)
        
        cax = plt.axes([horizontal_position[l], vertical_position, width, height])
        cbar = fig.colorbar(im, cax=cax, label=cbar_labels[l], orientation='horizontal')
        ax[l].set_xlabel(r'${\rm log }\delta$')
        ax[l].annotate(plot_lines[l], xy=(x[l], 0.85), xycoords='axes fraction', fontsize=13, bbox=dict(boxstyle="round", fc="w", lw=0.75))
        
        if l in [0, 1]:
            ax[l].set_xticks(range(-1, 7))
        elif l == 2:
            ax[l].set_xticks(range(-1, 8))

    ax[0].set_ylabel(r'${\rm log } (T / {\rm K})$')

    fig.subplots_adjust(wspace=0.,)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_no_uvb_deltaTN_{lines[0]}_{lines[1]}_{lines[2]}_chisqion_all_r200.png')
    plt.close()
