import h5py
import numpy as np
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    log_frad = sys.argv[4] # can be 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    # possible ions to choose:
    lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206', 'CIV1548', 'OVI1031']
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']
    vel_range = 600. #km/s
    bin_size = 6. # km/s 
    c = 2.98e8 # km/s
   
    spectra_dir = f'/disk04/sapple/data/satellites/{model}_{wind}_{snap}/log_frad_{log_frad}/'
    sample_dir = f'/disk04/sapple/data/samples/' 

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    for line in lines:

        results_file = f'/disk04/sapple/data/satellites/results/{model}_{wind}_{snap}_{log_frad}log_frad_ew_{line}.h5'

        if os.path.isfile(results_file):
            with h5py.File(results_file, 'r') as f:
                all_keys = list(f.keys())
        else:
            all_keys = []

        for i in range(len(fr200)):

            if not (f'ew_wave_{fr200[i]}r200' in all_keys) & (f'LOS_pos_{fr200[i]}r200' in all_keys):

                all_ew = np.zeros((len(gal_ids), len(orients)))
                all_los = np.zeros((len(gal_ids) * len(orients), 2))

                for j, gal in enumerate(gal_ids):
            
                    for o, orient in enumerate(orients):

                        spec_name = f'sample_galaxy_{gal_ids[j]}_{line}_{orient}_{fr200[i]}r200'
                        spec = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

                        vel_mask = (spec['velocities'] < spec['gal_velocity_pos'] + vel_range) & (spec['velocities'] > spec['gal_velocity_pos'] - vel_range)
                        flux = spec['fluxes'][vel_mask]
                        pixel_size = spec['wavelengths'][1] - spec['wavelengths'][0]
                        all_ew[j][o] = equivalent_width(flux, pixel_size)
                        all_los[j*o+o] = spec['LOS_pos'][:2]

                with h5py.File(results_file, 'a') as hf:
                    if not f'ew_wave_{fr200[i]}r200' in hf.keys():
                        hf.create_dataset(f'ew_wave_{fr200[i]}r200', data=np.array(all_ew))
                    if not f'LOS_pos_{fr200[i]}r200' in hf.keys():
                        hf.create_dataset(f'LOS_pos_{fr200[i]}r200', data=np.array(all_los))

