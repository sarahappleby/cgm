import h5py
import numpy as np
import pygad as pg
import sys
from generate_spectra import read_spectrum


def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr


def get_total_column_density(fit_logN, fit_dlogN):
    totalN = np.sum(10**(fit_logN))
    d_totalN = np.sqrt(np.sum(fit_dlogN**2.) * (np.log(10.)**2.) * (totalN**2.))
    return convert_to_log(totalN, d_totalN)


def exclude_lines_outside_window(lines, gal_vpos, vel_range, lambda_rest, z):
    delete_lines = []
    for l in range(len(lines['fit_l'])):
        wavelength = lines['fit_l'][l]
        velocity = wave_to_vel(wavelength, lambda_rest, z)
        if not (velocity < gal_vpos+vel_range) & (velocity > gal_vpos-vel_range):
            delete_lines.append(l)
    for k in lines.keys():
        lines[k] = np.delete(lines[k], delete_lines)
    return lines


def wave_to_vel(wave, lambda_rest, z):
        c = 2.99792e5 # km/s
        return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)


if __name__ == '__main__':

    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    """
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    """

    if snap == '151':
        z = 0.

    vel_range = 600 # km/s 
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    results_file = f'data/normal/results/fit_column_densities_{line}.h5'
    rchisq_file = f'data/normal/results/fit_max_rchisq_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_totalN = np.zeros((len(gal_ids), len(orients)))
    all_dtotalN = np.zeros((len(gal_ids), len(orients)))
    all_max_rchisq = np.zeros((len(gal_ids), len(orients)))

    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spectrum = read_spectrum(f'{spectra_dir}{spec_name}.h5')

            spectrum['lines'] = exclude_lines_outside_window(spectrum['lines'], spectrum['gal_velocity_pos'], vel_range, spectrum['lambda_rest'], z)
            all_totalN[i][o], all_dtotalN[i][o] = get_total_column_density(spectrum['lines']['fit_logN'], spectrum['lines']['fit_dlogN'])
           
            if len(spectrum['lines']['fit_Chisq']) > 0.:
                all_max_rchisq[i][o] = np.nanmax(spectrum['lines']['fit_Chisq'])
            else:
                all_max_rchisq[i][o] = -9999.

    with h5py.File(results_file, 'a') as hf:
        if not f'log_totalN_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_totalN_{fr200}r200', data=np.array(all_totalN))
        if not f'log_dtotalN_{fr200}r200' in hf.keys():
            hf.create_dataset(f'log_dtotalN_{fr200}r200', data=np.array(all_dtotalN))
    
    with h5py.File(rchisq_file, 'a') as hf:
        if not f'max_rchisq_{fr200}r200' in hf.keys():
            hf.create_dataset(f'max_rchisq_{fr200}r200', data=np.array(all_max_rchisq))

    """
    1) read in the list of galaxies
    2) for each sample galaxy:
        3) for each impact parameter:
            4) for each line of sight:
                5) find the overall absorption measures for each ion, i.e. combine the absorption features
                6) add these to a database
   
    one file per impact parameter
    file name: model_wind_snap_fr200_absorption.h5
    for each parameter in the file:
        2d array with shape (number of galaxies, number of lines of sight)

    """
