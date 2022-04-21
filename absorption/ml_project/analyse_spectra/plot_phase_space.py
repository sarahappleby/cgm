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

    cmap = plt.get_cmap('Greys')
    cmap = truncate_colormap(cmap, 0.0, .6)

    lines = ["H1215", "MgII2796", "CII1334"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$']
    cbar_labels = [r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm MgII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$']
    N_min = [12., 11., 12.]
    x = [0.79, 0.75, 0.78]
    #chisq_lim = [4.5, 63.1, 20.0]
    chisq_lim = [4., 50., 15.8]

    lines = ["SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    cbar_labels = [r'${\rm log }(N\ {\rm SiIII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CIV} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm OVI} / {\rm cm}^{-2})$']
    N_min = [11., 12., 12.]
    x = [0.765, 0.765, 0.77]
    #chisq_lim = [70.8, 15.8, 4.5]
    chisq_lim = [39.8, 8.9, 4.5]

    #width = 0.258
    #height = 0.015
    #vertical_position = 0.95
    #horizontal_position = [0.125, 0.3833, 0.6416]

    width = 0.208
    height = 0.015
    vertical_position = 0.95
    horizontal_position = [0.15, 0.4088, 0.666]
    
    xticks = [[-1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]

    deltath = 2.046913
    Tth = 5.

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    ssfr_labels = ['All', 'Star forming', 'Green valley', 'Quenched']

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    phase_space_file = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}_phase_space.h5'
    with h5py.File(phase_space_file, 'r') as hf:
        rho_overdensity_temp_hist2d = hf['rho_delta_temp'][:]
        rho_overdensity_bins = hf['rho_delta_bins'][:]
        temp_bins = hf['temp_bins'][:]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    # inner outer cgm
    fig, ax = plt.subplots(len(inner_outer), len(lines), figsize=(15, 7.1), sharey='row', sharex='col')
    
    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(inner_outer)):

            all_T = []
            all_rho = []
            all_N = []
            all_chisq = []
            all_ids = []

            for j in range(len(inner_outer[i])):

                with h5py.File(results_file, 'r') as hf:
                    all_T.extend(hf[f'log_T_{inner_outer[i][j]}r200'][:])
                    all_rho.extend(hf[f'log_rho_{inner_outer[i][j]}r200'][:])
                    all_N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                    all_ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])

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

            ax[i][l].imshow(np.log10(rho_overdensity_temp_hist2d), extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]), 
                            cmap=cmap)

            ax[i][l].axhline(Tth, c='k', ls=':', lw=1)
            ax[i][l].axvline(deltath, c='k', ls=':', lw=1)

            if line == 'H1215':
                im = ax[i][l].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=16)
            else:
                im = ax[i][l].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=15)
            
            ax[i][l].set_xlim(-1, 5)
            ax[i][l].set_ylim(3, 7)
        
            if i == 0:
                cax = plt.axes([horizontal_position[l], vertical_position, width, height])
                cbar = fig.colorbar(im, cax=cax, label=cbar_labels[l], orientation='horizontal')
            if l == len(lines) -1:
                if i == 0:
                    ax[i][l].annotate(rho_labels[i], xy=(0.71, 0.87), xycoords='axes fraction', fontsize=13, bbox=dict(boxstyle="round", fc="w", lw=0.75))
                elif i == 1:
                    ax[i][l].annotate(rho_labels[i], xy=(0.7, 0.87), xycoords='axes fraction', fontsize=13, bbox=dict(boxstyle="round", fc="w", lw=0.75))
            if i == 1:
                ax[i][l].set_xlabel(r'${\rm log }\delta$')
                ax[i][l].annotate(plot_lines[l], xy=(x[l], 0.09), xycoords='axes fraction', fontsize=13, bbox=dict(boxstyle="round", fc="w", lw=0.75))
            if l == 0:
                ax[i][l].set_ylabel(r'${\rm log } (T / {\rm K})$')

        if l in [0, 1]:
            ax[1][l].set_xticks(range(-1, 5))
        elif l == 2:
            ax[1][l].set_xticks(range(-1, 6))


    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN_{lines[0]}_{lines[1]}_{lines[2]}_chisqion.png')
    plt.close()
