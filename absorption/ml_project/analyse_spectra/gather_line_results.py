import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import interpolate
import pygad as pg
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

def get_interp_conditions(wavelengths, lines, quantity):
    wave_boxsize = wavelengths[-1] - wavelengths[0]
    for i in range(len(lines)):
        if lines[i] < np.min(wavelengths):
            lines[i] += wave_boxsize
        elif lines[i] > np.max(wavelengths):
            lines[i] -= wave_boxsize

    model = interpolate.interp1d(wavelengths, quantity)
    return model(lines)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    if snap == '151':
        redshift = 0.

    vel_range = 600. # km/s
    lambda_rest = float(pg.analysis.absorption_spectra.lines[line]['l'].split()[0])
    wave_range = float(vel_to_wave(vel_range, lambda_rest, redshift)) - lambda_rest

    """
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    """

    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_rho = []
    all_T = []
    all_Z = []
    all_Nspec = []
    all_vpec = []

    all_N = []
    all_b = []
    all_l = []
    all_ew = []
    all_chisq = []
    all_ids = []

    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

            if len(spectrum['line_list']['N']) > 0.:
               
                wave_boxsize = spectrum['wavelengths'][-1] - spectrum['wavelengths'][0]
                gal_wave_pos = vel_to_wave(spectrum['gal_velocity_pos'], lambda_rest, redshift)

                for j in range(len(spectrum['line_list']['l'])):
                    if spectrum['line_list']['l'][j] < np.min(spectrum['wavelengths']):
                        spectrum['line_list']['l'][j] += wave_boxsize
                    elif spectrum['line_list']['l'][j] > np.max(spectrum['wavelengths']):
                        spectrum['line_list']['l'][j] -= wave_boxsize

                line_mask = (spectrum['line_list']['l'] > gal_wave_pos - wave_range) & (spectrum['line_list']['l'] < gal_wave_pos + wave_range)

                all_rho.extend(get_interp_conditions(spectrum['wavelengths'], spectrum['line_list']['l'][line_mask], np.log10(spectrum['phys_density'])))
                all_T.extend(get_interp_conditions(spectrum['wavelengths'], spectrum['line_list']['l'][line_mask], np.log10(spectrum['temperature'])))
                all_Z.extend(get_interp_conditions(spectrum['wavelengths'], spectrum['line_list']['l'][line_mask], spectrum['metallicity']))
                all_Nspec.extend(get_interp_conditions(spectrum['wavelengths'], spectrum['line_list']['l'][line_mask], np.log10(spectrum['col_density'])))
                all_vpec.extend(get_interp_conditions(spectrum['wavelengths'], spectrum['line_list']['l'][line_mask], spectrum['vpec']))

                all_chisq.extend(spectrum['line_list']['Chisq'][line_mask])
                all_N.extend(spectrum['line_list']['N'][line_mask])
                all_b.extend(spectrum['line_list']['b'][line_mask])
                all_l.extend(spectrum['line_list']['l'][line_mask])
                all_ew.extend(spectrum['line_list']['EW'][line_mask])
                all_ids.extend([gal_ids[i]] * len(spectrum['line_list']['N'][line_mask]))
    
    with h5py.File(results_file, 'a') as hf:
        if not f'log_rho_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_rho_{fr200}r200', data=np.array(all_rho))
        if not f'log_T_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_T_{fr200}r200', data=np.array(all_T))
        if not f'log_Z_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_Z_{fr200}r200', data=np.array(all_Z))
        if not f'log_Nspec_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_Nspec_{fr200}r200', data=np.array(all_Nspec))
        if not f'vpec_{fr200}r200' in hf.keys():
            hf.create_dataset(f'vpec_{fr200}r200', data=np.array(all_vpec))
        if not f'log_N_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_N_{fr200}r200', data=np.array(all_N))
        if not f'b_{fr200}r200' in hf.keys():
            hf.create_dataset(f'b_{fr200}r200', data=np.array(all_b))
        if not f'l_{fr200}r200' in hf.keys():
            hf.create_dataset(f'l_{fr200}r200', data=np.array(all_l))
        if not f'ew_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ew_{fr200}r200', data=np.array(all_ew))
        if not f'chisq_{fr200}r200' in hf.keys():
            hf.create_dataset(f'chisq_{fr200}r200', data=np.array(all_chisq))
        if not f'ids_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ids_{fr200}r200', data=np.array(all_ids))
