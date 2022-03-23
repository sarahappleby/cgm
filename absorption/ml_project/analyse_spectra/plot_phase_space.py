import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
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
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    ion_mass = np.array([pg.UnitArr(pg.analysis.absorption_spectra.lines[line]['atomwt']) * pg.physics.m_u for line in lines])
    chisq_lim = 2.5
    N_min = [12., 11., 12., 11., 12., 12.]
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]
    
    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    phase_space_file = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}_phase_space.h5'
    with h5py.File(phase_space_file, 'r') as hf:
        rho_overdensity_temp_hist2d = hf['rho_overdensity_temp'][:]
        nh_temp_hist2d = hf['nh_temp'][:]
        rho_overdensity_bins = hf['rho_overdensity_bins'][:]
        temp_bins = hf['temp_bins'][:]
        n_bins = hf['n_bins'][:]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    width = 0.007
    height = 0.1132
    vertical_position = [0.76, 0.632, 0.504, 0.373, 0.247, 0.1175]
    horizontal_position = 0.9
    
    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(14, 13), sharey='row', sharex='col')
    
    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_rho = hf[f'log_rho_{fr200[i]}r200'][:]
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min[l]) * (all_chisq < chisq_lim)
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
            all_ids = all_ids[mask]
            all_N = all_N[mask]

            ax[l][i].imshow(rho_overdensity_temp_hist2d, extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]), cmap='Greys')

            if line == 'H1215':
                im = ax[l][i].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=16)
            else:
                im = ax[l][i].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=15)
            ax[l][i].set_xlim(-1, 5)
            ax[l][i].set_ylim(3, 7)

            #if i == len(fr200) -1:
            #    ax_divider = make_axes_locatable(ax[l][-1])
            #    cax = ax_divider.append_axes("right", size="7%", pad="2%")
            #    fig.colorbar(im, cax=cax, label=r'${\rm log }(N / {\rm cm}^{-2})$')
            if i == len(fr200) -1:
                cax = plt.axes([horizontal_position, vertical_position[l], width, height])
                fig.colorbar(im, cax=cax, label=r'${\rm log }(N / {\rm cm}^{-2})$')
                ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'${\rm log }\Delta$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log } (T / {\rm K})$')

    #plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN_ugh.png')
    plt.close()
   
    """
    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_rho = hf[f'log_rho_{fr200[i]}r200'][:]
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min[l]) * (all_chisq < chisq_lim)
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
            all_ids = all_ids[mask]
            all_N = all_N[mask]

            ax[l][i].imshow(rho_overdensity_temp_hist2d, extent=(rho_overdensity_bins[0], rho_overdensity_bins[-1], temp_bins[0], temp_bins[-1]), cmap='Greys')

            if line == 'H1215':
                im = ax[l][i].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=16)
            else:
                im = ax[l][i].scatter(all_delta_rho, all_T, c=all_N, cmap='magma', s=1, vmin=N_min[l], vmax=15)
            ax[l][i].set_xlim(-1, 5)
            ax[l][i].set_ylim(3, 7)

            if i == len(fr200) -1:
                fig.colorbar(im, ax=ax[l][-1], label=r'${\rm log }(N / {\rm cm}^{-2})$')
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'${\rm log }\Delta$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log } (T / {\rm K})$')
                ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_deltaTN.png')
    plt.close()

    fig, ax = plt.subplots(len(lines), len(fr200)+1, figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        all_los_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_los_rhoTZ_{line}.h5'
        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(all_los_file, 'r') as hf:
                all_Z = hf[f'los_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'los_T_{fr200[i]}r200'][:]
                all_rho = np.log10(hf[f'los_rho_{fr200[i]}r200'][:]) - np.log10(ion_mass[l])

            with h5py.File(results_file, 'r') as hf:
                all_Z = hf[f'log_Z_{fr200[i]}r200'][:] - np.log10(zsolar[l])
                all_T = hf[f'log_T_{fr200[i]}r200'][:]
                all_rho = hf[f'log_rho_{fr200[i]}r200'][:]
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min[l]) * (all_chisq < chisq_lim)
            all_Z = all_Z[mask]
            all_T = all_T[mask]
            all_n = np.log10(10**all_Z * 10**all_rho[mask]) - np.log10(ion_mass[l])
            all_ids = all_ids[mask]
            all_N = all_N[mask]

            ax[l][i].imshow(nh_temp_hist2d, extent=(n_bins[0], n_bins[-1], temp_bins[0], temp_bins[-1]), cmap='Greys')

            im = ax[l][i].scatter(all_n, all_T, c=all_Z, cmap='magma', s=1, vmin=-1., vmax=0.5)
            ax[l][i].set_xlim(-6, 0)
            ax[l][i].set_ylim(3, 7)

            if i == len(fr200) -1:
                fig.colorbar(im, ax=ax[l][-1], label=r'${\rm log} (Z / Z_{\odot})$')
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'${\rm log }(n / {\rm cm}^{-3})$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log } (T / {\rm K})$')
                ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_nTZ.png')
    plt.close()
    """
    
'''
FYI: Gizmo metallicity structure
All.SolarAbundances[0]=0.0134;        // all metals (by mass); present photospheric abundances from Asplund et al. 2009 (Z=0.0134, proto-solar=0.0142) in notes;
All.SolarAbundances[1]=0.2485;    // He  (10.93 in units where log[H]=12, so photospheric mass fraction -> Y=0.2485 [Hydrogen X=0.7381]; Anders+Grevesse Y=0.2485, X=0.7314)
All.SolarAbundances[2]=2.38e-3; // C   (8.43 -> 2.38e-3, AG=3.18e-3)
All.SolarAbundances[3]=0.70e-3; // N   (7.83 -> 0.70e-3, AG=1.15e-3)
All.SolarAbundances[4]=5.79e-3; // O   (8.69 -> 5.79e-3, AG=9.97e-3)
All.SolarAbundances[5]=1.26e-3; // Ne  (7.93 -> 1.26e-3, AG=1.72e-3)
All.SolarAbundances[6]=7.14e-4; // Mg  (7.60 -> 7.14e-4, AG=6.75e-4)
All.SolarAbundances[7]=6.71e-3; // Si  (7.51 -> 6.71e-4, AG=7.30e-4)
All.SolarAbundances[8]=3.12e-4; // S   (7.12 -> 3.12e-4, AG=3.80e-4)
All.SolarAbundances[9]=0.65e-4; // Ca  (6.34 -> 0.65e-4, AG=0.67e-4)
All.SolarAbundances[10]=1.31e-3; // Fe (7.50 -> 1.31e-3, AG=1.92e-3)
'''

