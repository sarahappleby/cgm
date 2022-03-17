import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import pygad as pg
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import wave_to_vel, vel_to_wave

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)
    
def get_outlying_line_mask(spectrum, gal_wave_pos, wave_range, wave_boxsize, chisq_lim):

    chisq_mask = spectrum['line_list']['Chisq'] < chisq_lim

    for j in range(len(spectrum['line_list']['l'])):
        if spectrum['line_list']['l'][j] < np.min(spectrum['wavelengths']):
            spectrum['line_list']['l'][j] += wave_boxsize
        elif spectrum['line_list']['l'][j] > np.max(spectrum['wavelengths']):
            spectrum['line_list']['l'][j] -= wave_boxsize

    line_mask = (spectrum['line_list']['l'] > gal_wave_pos - wave_range) & (spectrum['line_list']['l'] < gal_wave_pos + wave_range)

    for key in list(spectrum['line_list'].keys()):
        spectrum['line_list'][key] = spectrum['line_list'][key][line_mask*chisq_mask]

    return spectrum


if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    fr200 = 1.0

    line_a = 'H1215'
    line_b = 'OVI1031'
    lambda_a = float(pg.UnitArr(pg.analysis.absorption_spectra.lines[line_a]['l']))
    lambda_b = float(pg.UnitArr(pg.analysis.absorption_spectra.lines[line_b]['l']))
    redshift = 0.
    vel_range = 600.
    wave_range_a = float(vel_to_wave(vel_range, lambda_a, redshift)) - lambda_a
    wave_range_b = float(vel_to_wave(vel_range, lambda_b, redshift)) - lambda_b

    chisq_lim = 2.5
    all_dv = np.arange(5., 105., 5.)

    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b}_{fr200}r200.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    for dv in all_dv:

        aligned_ids = []
        aligned_ew_a = []
        aligned_v_a = []
        aligned_b_a = []
        aligned_N_a = []
        aligned_ew_b = []
        aligned_v_b = []
        aligned_b_b = []
        aligned_N_b = []

        for i in range(len(gal_ids)):
            for o, orient in enumerate(orients):
                spectrum_a = read_h5_into_dict(f'{spectra_dir}sample_galaxy_{gal_ids[i]}_{line_a}_{orient}_{fr200}r200.h5')
                spectrum_b = read_h5_into_dict(f'{spectra_dir}sample_galaxy_{gal_ids[i]}_{line_b}_{orient}_{fr200}r200.h5')

                if (len(spectrum_a['line_list']['l']) > 0) & (len(spectrum_b['line_list']['l']) > 0):
                    wave_boxsize_a = spectrum_a['wavelengths'][-1] - spectrum_a['wavelengths'][0]
                    wave_boxsize_b = spectrum_b['wavelengths'][-1] - spectrum_b['wavelengths'][0]
                    gal_wave_pos_a = vel_to_wave(spectrum_a['gal_velocity_pos'], lambda_a, redshift)
                    gal_wave_pos_b = vel_to_wave(spectrum_b['gal_velocity_pos'], lambda_b, redshift)

                    spectrum_a = get_outlying_line_mask(spectrum_a, gal_wave_pos_a, wave_range_a, wave_boxsize_a, chisq_lim)                
                    spectrum_b = get_outlying_line_mask(spectrum_b, gal_wave_pos_b, wave_range_b, wave_boxsize_b, chisq_lim)
                
                    vels_a = np.array(wave_to_vel(spectrum_a['line_list']['l'], lambda_a, redshift))
                    vels_b = np.array(wave_to_vel(spectrum_b['line_list']['l'], lambda_b, redshift))
            
                    for j in range(len(vels_b)):
                        vel_sep = vels_a - vels_b[j]
                        vel_mask = np.abs(vel_sep) < dv
                        if len(vel_sep[vel_mask]) > 0.:
                        
                            closest = np.argmin(np.abs(vel_sep[vel_mask]))

                            aligned_ids.append(gal_ids[i])
                    
                            aligned_ew_a.append(spectrum_a['line_list']['EW'][vel_mask][closest])
                            aligned_v_a.append(vels_a[vel_mask][closest])
                            aligned_b_a.append(spectrum_a['line_list']['b'][vel_mask][closest])
                            aligned_N_a.append(spectrum_a['line_list']['N'][vel_mask][closest])

                            aligned_ew_b.append(spectrum_b['line_list']['EW'][j])
                            aligned_v_b.append(vels_b[j])
                            aligned_b_b.append(spectrum_b['line_list']['b'][j])
                            aligned_N_b.append(spectrum_b['line_list']['N'][j])
    
        with h5py.File(results_file, 'a') as hf:
            if not f'ids_{dv}kms' in hf.keys():
                hf.create_dataset(f'ids_{dv}kms', data=np.array(aligned_ids))
            if not f'{line_a}_ew_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_a}_ew_{dv}kms', data=np.array(aligned_ew_a))
            if not f'{line_a}_v_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_a}_v_{dv}kms', data=np.array(aligned_v_a))
            if not f'{line_a}_b_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_a}_b_{dv}kms', data=np.array(aligned_b_a))
            if not f'{line_a}_log_N_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_a}_log_N_{dv}kms', data=np.array(aligned_N_a))
            if not f'{line_b}_ew_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_b}_ew_{dv}kms', data=np.array(aligned_ew_b))
            if not f'{line_b}_v_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_b}_v_{dv}kms', data=np.array(aligned_v_b))
            if not f'{line_b}_b_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_b}_b_{dv}kms', data=np.array(aligned_b_b))
            if not f'{line_b}_log_N_{dv}kms' in hf.keys():
                hf.create_dataset(f'{line_b}_log_N_{dv}kms', data=np.array(aligned_N_b))
