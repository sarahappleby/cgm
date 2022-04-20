import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
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

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    x = [0.75, 0.69, 0.73, 0.705, 0.71, 0.71]
    cbar_ticks = [[12, 13, 14, 15, 16], [11, 12, 13, 14], [12, 13, 14], [11, 12, 13, 14], [12, 13, 14], [12, 13, 14],]
    #chisq_lim = [4.5, 63.1, 20.0, 70.8, 15.8, 4.5] limits with old fitting procedure
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]

    width = 0.007
    height = 0.1283
    vertical_position = [0.76, 0.632, 0.504, 0.373, 0.247, 0.1175]
    vertical_position = [0.7516, 0.623, 0.495, 0.366, 0.238, 0.11]
    horizontal_position = 0.9

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    rho_labels = ['Inner CGM', 'Outer CGM']
    ssfr_labels = ['Star forming', 'Green valley', 'Quenched']

    N_min = [12., 11., 12., 11., 12., 12.]
    
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

    # ssfr split, all fr200
    fig, ax = plt.subplots(len(lines), 3, figsize=(9.7, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_Z = []
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

        idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
        all_mass = mass[idx]
        all_ssfr = ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        for i in range(3):
            ax[l][i].imshow(np.log10(rho_overdensity_temp_hist2d), extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]),
                            cmap=cmap)
        
        if line == 'H1215':
            im = ax[l][0].scatter(all_delta_rho[sf_mask], all_T[sf_mask], c=all_N[sf_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
            im = ax[l][1].scatter(all_delta_rho[gv_mask], all_T[gv_mask], c=all_N[gv_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
            im = ax[l][2].scatter(all_delta_rho[q_mask], all_T[q_mask], c=all_N[q_mask], cmap='magma', s=1, vmin=N_min[l], vmax=16)
        else:
            im = ax[l][0].scatter(all_delta_rho[sf_mask], all_T[sf_mask], c=all_N[sf_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
            im = ax[l][1].scatter(all_delta_rho[gv_mask], all_T[gv_mask], c=all_N[gv_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
            im = ax[l][2].scatter(all_delta_rho[q_mask], all_T[q_mask], c=all_N[q_mask], cmap='magma', s=1, vmin=N_min[l], vmax=15)
        
        for i in range(3):
            ax[l][i].set_xlim(-1, 5)
            ax[l][i].set_ylim(3, 7)

        cax = plt.axes([horizontal_position, vertical_position[l], width, height])
        cbar = fig.colorbar(im, cax=cax, label=r'${\rm log }(N / {\rm cm}^{-2})$')
        cbar.set_ticks(cbar_ticks[l])
        ax[l][0].annotate(plot_lines[l], xy=(x[l], 0.85), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))
        
        if l == 0:
            for i in range(3):
                ax[l][i].set_title(ssfr_labels[i])
        if l == len(lines)-1:
            for i in range(3):
                ax[l][i].set_xlabel(r'${\rm log }\Delta$')
            ax[l][0].set_yticks([3, 4, 5, 6, 7])
        else:
            ax[l][0].set_yticks([4, 5, 6, 7])

        ax[l][0].set_ylabel(r'${\rm log } (T / {\rm K})$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN_ssfr_split_chisqion.png')
    plt.close()

