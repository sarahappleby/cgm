import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import interpolate
import sys
import pygad as pg
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import wave_to_vel, vel_to_wave

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)
    
def mask_outlying_lines(spectrum, vel_range, vel_boxsize, chisq_lim):

    chisq_mask = spectrum['line_list']['Chisq'] < chisq_lim

    """
    for j in range(len(spectrum['line_list']['v'])):
        if spectrum['line_list']['v'][j] < 0:
            spectrum['line_list']['v'][j] += vel_boxsize
        elif spectrum['line_list']['v'][j] > vel_boxsize:
            spectrum['line_list']['v'][j] -= wave_boxsize
    """

    line_mask = (spectrum['line_list']['v'] > spectrum['gal_velocity_pos'] - vel_range) & (spectrum['line_list']['v'] < spectrum['gal_velocity_pos'] + vel_range)

    for key in list(spectrum['line_list'].keys()):
        spectrum['line_list'][key] = spectrum['line_list'][key][line_mask*chisq_mask]

    return spectrum


def get_interp_conditions(wavelengths, lines, quantity):
    wave_boxsize = wavelengths[-1] - wavelengths[0]
    for i in range(len(lines)):
        if lines[i] < np.min(wavelengths):
            lines[i] += wave_boxsize
        elif lines[i] > np.max(wavelengths):
            lines[i] -= wave_boxsize

    model = interpolate.interp1d(wavelengths, quantity)
    return float(model(lines))


if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    fr200 = sys.argv[1]

    line_a = 'CII1334'
    line_b = 'OVI1031'
    
    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
 
    vel_range = 600.
    vel_boxsize = 10000.
    chisq_lim = 2.5
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    aligned_ids = []
    aligned_dv = []

    aligned_ew_a = []
    aligned_v_a = []
    aligned_b_a = []
    aligned_N_a = []
    aligned_rho_a = []
    aligned_T_a = []
    aligned_Z_a = []

    aligned_ew_b = []
    aligned_v_b = []
    aligned_b_b = []
    aligned_N_b = []
    aligned_rho_b = []
    aligned_T_b = []
    aligned_Z_b = []

    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spectrum_a = read_h5_into_dict(f'{spectra_dir}sample_galaxy_{gal_ids[i]}_{line_a}_{orient}_{fr200}r200.h5')
            spectrum_b = read_h5_into_dict(f'{spectra_dir}sample_galaxy_{gal_ids[i]}_{line_b}_{orient}_{fr200}r200.h5')

            if not (len(spectrum_a['line_list']['l']) > 0) & (len(spectrum_b['line_list']['l']) > 0):
                    continue

            spectrum_a['line_list']['v'] = np.array(wave_to_vel(spectrum_a['line_list']['l'], spectrum_a['lambda_rest'], redshift))
            spectrum_b['line_list']['v'] = np.array(wave_to_vel(spectrum_b['line_list']['l'], spectrum_b['lambda_rest'], redshift))
                    
            spectrum_a = mask_outlying_lines(spectrum_a, vel_range, vel_boxsize, chisq_lim)                
            spectrum_b = mask_outlying_lines(spectrum_b, vel_range, vel_boxsize, chisq_lim)

            if not (len(spectrum_a['line_list']['l']) > 0) & (len(spectrum_b['line_list']['l']) > 0):
                continue

            for j in range(len(spectrum_b['line_list']['v'])):
                        
                dv = np.ones(len(spectrum_a['line_list']['v'])) * np.nan

                for k in range(len(spectrum_a['line_list']['v'])):

                    if spectrum_b['line_list']['v'][j] > spectrum_a['line_list']['v'][k]:
                        dv[k] = spectrum_b['line_list']['v'][j] - spectrum_b['line_list']['b'][j] -  (spectrum_a['line_list']['v'][k] -  spectrum_a['line_list']['b'][k])
                            
                    elif spectrum_b['line_list']['v'][j] < spectrum_a['line_list']['v'][k]:
                        dv[k] = spectrum_a['line_list']['v'][k] - spectrum_a['line_list']['b'][k] -  (spectrum_b['line_list']['v'][j] -  spectrum_b['line_list']['b'][j])

                closest = np.argmin(np.abs(dv))
                if dv[closest] < 0.:
                    dv[closest] = 0.
                            
                aligned_dv.append(dv[closest])
                aligned_ids.append(gal_ids[i])
                    
                aligned_ew_a.append(spectrum_a['line_list']['EW'][closest])
                aligned_v_a.append(spectrum_a['line_list']['v'][closest])
                aligned_b_a.append(spectrum_a['line_list']['b'][closest])
                aligned_N_a.append(spectrum_a['line_list']['N'][closest])
                aligned_rho_a.append(get_interp_conditions(spectrum_a['wavelengths'], [spectrum_a['line_list']['l'][closest]], 
                                     np.log10(spectrum_a['phys_density'])))
                aligned_T_a.append(get_interp_conditions(spectrum_a['wavelengths'], [spectrum_a['line_list']['l'][closest]],
                                   np.log10(spectrum_a['temperature'])))
                aligned_Z_a.append(get_interp_conditions(spectrum_a['wavelengths'], [spectrum_a['line_list']['l'][closest]],
                                   spectrum_a['metallicity']))
                            

                aligned_ew_b.append(spectrum_b['line_list']['EW'][j])
                aligned_v_b.append(spectrum_b['line_list']['v'][j])
                aligned_b_b.append(spectrum_b['line_list']['b'][j])
                aligned_N_b.append(spectrum_b['line_list']['N'][j])
                aligned_rho_b.append(get_interp_conditions(spectrum_b['wavelengths'], [spectrum_b['line_list']['l'][j]],
                                     np.log10(spectrum_b['phys_density'])))
                aligned_T_b.append(get_interp_conditions(spectrum_b['wavelengths'], [spectrum_b['line_list']['l'][j]],
                                   np.log10(spectrum_b['temperature'])))
                aligned_Z_b.append(get_interp_conditions(spectrum_b['wavelengths'], [spectrum_b['line_list']['l'][j]],
                                   spectrum_b['metallicity']))


    with h5py.File(results_file, 'a') as hf:
        if not f'ids_{fr200}' in hf.keys():
            hf.create_dataset(f'ids_{fr200}', data=np.array(aligned_ids))
        if not f'dv_{fr200}' in hf.keys():
            hf.create_dataset(f'dv_{fr200}', data=np.array(aligned_dv))
    

        if not f'{line_a}_ew_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_ew_{fr200}', data=np.array(aligned_ew_a))
        if not f'{line_a}_v_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_v_{fr200}', data=np.array(aligned_v_a))
        if not f'{line_a}_b_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_b_{fr200}', data=np.array(aligned_b_a))
        if not f'{line_a}_log_N_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_log_N_{fr200}', data=np.array(aligned_N_a))
        if not f'{line_a}_log_rho_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_log_rho_{fr200}', data=np.array(aligned_rho_a))
        if not f'{line_a}_log_T_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_log_T_{fr200}', data=np.array(aligned_T_a))
        if not f'{line_a}_log_Z_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_a}_log_Z_{fr200}', data=np.array(aligned_Z_a))
            
        if not f'{line_b}_ew_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_ew_{fr200}', data=np.array(aligned_ew_b))
        if not f'{line_b}_v_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_v_{fr200}', data=np.array(aligned_v_b))
        if not f'{line_b}_b_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_b_{fr200}', data=np.array(aligned_b_b))
        if not f'{line_b}_log_N_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_log_N_{fr200}', data=np.array(aligned_N_b))
        if not f'{line_b}_log_rho_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_log_rho_{fr200}', data=np.array(aligned_rho_b))
        if not f'{line_b}_log_T_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_log_T_{fr200}', data=np.array(aligned_T_b))
        if not f'{line_b}_log_Z_{fr200}' in hf.keys():
            hf.create_dataset(f'{line_b}_log_Z_{fr200}', data=np.array(aligned_Z_b))
