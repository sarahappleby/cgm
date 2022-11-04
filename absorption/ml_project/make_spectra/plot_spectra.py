# Plot the spectra for each galaxy

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import sys
sys.path.append('../make_spectra/')
from utils import read_h5_into_dict
from physics import wave_to_vel, tau_to_flux
from spectrum import Spectrum

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif', size=13)
plt.rc('font', size=12)

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = 151
    
    orient = '135_deg'
    fr200 = 0.25
    vel_range = 600.
    redshift = 0

    gal_id = 1464

    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    chisq_strings = [r'$\chi^2_{r, 1}$', r'$\chi^2_{r, 2}$', r'$\chi^2_{r, 3}$', r'$\chi^2_{r, 4}$']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        spec_name = f'sample_galaxy_{gal_id}_{line}_{orient}_{fr200}r200'
        spec = Spectrum(f'{spectra_dir}{spec_name}.h5')
        
        spec.prepare_spectrum(vel_range=vel_range)
        regions_l, regions_i = pg.analysis.find_regions(spec.waves_fit, spec.fluxes_fit, spec.noise, min_region_width=2, extend=True)
        regions_v = np.array(wave_to_vel(regions_l, spec.lambda_rest, redshift)) - spec.gal_velocity_pos

        for k in range(len(regions_v)):
            ax[i][j].fill_between(regions_v[k], 0, 1.2, color='#E6E6E6')

            ax[i][j].axvline(regions_v[k][0], ymin=0.96, ls='-', c='darkgray', lw=1)
            ax[i][j].axvline(regions_v[k][1], ymin=0.96, ls='-', c='darkgray', lw=1)

        ax[i][j].plot(spec.velocities - spec.gal_velocity_pos, spec.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        tau_model = np.zeros(len(spec.wavelengths))
        for k in range(len(spec.line_list['N'])):
            
            p = np.array([spec.line_list['N'][k], spec.line_list['b'][k], spec.line_list['l'][k]])
            line_tau_model = pg.analysis.model_tau(spec.ion_name, p, spec.wavelengths)
            
            ax[i][j].plot(spec.velocities - spec.gal_velocity_pos, tau_to_flux(line_tau_model), c='tab:pink', alpha=0.5, lw=1, ls='--')
            
            tau_model += line_tau_model

        spec.flux_model = tau_to_flux(tau_model)
        ax[i][j].plot(spec.velocities - spec.gal_velocity_pos, spec.flux_model, label='model', c='tab:pink', ls='-', lw=2)

        ax[i][j].set_ylim(0, 1.15)
        ax[i][j].set_xlim(-vel_range, vel_range)

        regions_mid = (regions_v[:][:, 1] - regions_v[:][:, 0]) * 0.5 + regions_v[:][:, 0]

        indexes = np.unique(spec.line_list['Chisq'], return_index=True)[1]
        chisq = np.around([spec.line_list['Chisq'][index] for index in sorted(indexes)], 2)
        for k in range(len(chisq)):
            ax[i][j].annotate(str(k+1), xy=(regions_mid[k] - 10, 1.03), xycoords='data',color='#3C887E')
        chisq_str = f'{plot_lines[lines.index(line)]}'
        for k in range(len(chisq)):
            chisq_str += f'\n{chisq_strings[k]} = {chisq[k]}'
        ax[i][j].annotate(chisq_str, xy=(0.05, 0.10), xycoords='axes fraction',
                          bbox=dict(boxstyle="round, pad=0.5", fc="w", lw=0.75, alpha=0.8, edgecolor='tab:grey'))
        #ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.05), xycoords='axes fraction')

        mask = spec.line_list['Chisq'] < chisq_lim[lines.index(line)]
        vels = np.array(wave_to_vel(spec.line_list['l'][mask], spec.lambda_rest, redshift)) 

        if line == 'H1215':
            ax[i][j].legend(loc=4)

        if line in ['H1215', 'CII1334', 'CIV1548']:
            ax[i][j].set_ylabel('Flux')
        if line in ['CIV1548', 'OVI1031']:
            ax[i][j].set_xlabel('Velocity (km/s)')

        if line == 'CIV1548':
            ax[i][j].set_xticks(np.arange(-600, 600, 200))

        j += 1
        if line in ['MgII2796', 'SiIII1206'] :
            i += 1
            j = 0

    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig(f'{plot_dir}sample_galaxy_{gal_id}_{orient}_{fr200}r200.pdf', format='pdf')
    plt.close()


