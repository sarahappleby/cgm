import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
from utils import read_h5_into_dict
from physics import wave_to_vel

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

cb_blue = '#5289C7'

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = 151
    
    orient = '180_deg'
    fr200 = 0.25
    vel_range = 600.
    chisq_lim = 2.5
    redshift = 0

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    for gal_id in gal_ids:

        fig, ax = plt.subplots(len(lines), 2, figsize=(10, 10), sharex='col')

        for i, line in enumerate(lines):

            spec_name = f'sample_galaxy_{gal_id}_{line}_{orient}_{fr200}r200'
            spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

            ax[i][0].plot(spectrum['velocities'], spectrum['fluxes'], c=cb_blue, ls='-', lw=1)
            ax[i][0].annotate(plot_lines[i], xy=(0.75, 0.05), xycoords='axes fraction')
            ax[i][0].set_xlim(spectrum['gal_velocity_pos'] - vel_range, spectrum['gal_velocity_pos'] + vel_range)
            ax[i][0].set_ylim(0, 1.1)
            ax[i][0].set_ylabel('Flux')

            ax[i][1].plot(spectrum['velocities'], np.log10(spectrum['taus']), c=cb_blue, ls='-', lw=1)
            ax[i][1].set_xlim(spectrum['gal_velocity_pos'] - vel_range, spectrum['gal_velocity_pos'] + vel_range)
            ax[i][1].set_ylim(-6, 1)
            ax[i][1].set_ylabel(r'$\tau$')

            mask = spectrum['line_list']['Chisq'] < chisq_lim
            vels = np.array(wave_to_vel(spectrum['line_list']['l'][mask], spectrum['lambda_rest'], redshift)) 

            for j in range(len(vels)):
                for k in range(2):
                    ax[i][k].axvline(vels[j], c='k', ls='-', lw=1)

        ax[i][0].set_xlabel('Velocity (km/s)')
        ax[i][1].set_xlabel('Velocity (km/s)')

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.)
        plt.savefig(f'align_plots/sample_galaxy_{gal_id}_{orient}_{fr200}r200_chisq_lim_{chisq_lim}.png')
        plt.close()


