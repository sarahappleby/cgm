import h5py
import numpy as np
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    # possible ions to choose:
    ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']
    vel_range = 600. #km/s
    bin_size = 6. # km/s 
    c = 2.98e8 # km/s
   
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/ew_{line}.h5'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/' 

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_ew = np.zeros((len(gal_ids), len(orients)))

    for i, gal in enumerate(gal_ids):
        
        for o, orient in enumerate(orients):

            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spec = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

            vel_mask = (spec['velocities'] < spec['gal_velocity_pos'] + vel_range) & (spec['velocities'] > spec['gal_velocity_pos'] - vel_range)
            flux = spec['fluxes'][vel_mask]
            pixel_size = spec['wavelengths'][1] - spec['wavelengths'][0]
            all_ew[i][o] = equivalent_width(flux, pixel_size)

    with h5py.File(results_file, 'a') as f:
        f.create_dataset(f'ew_wave_{fr200}r200', data=np.array(all_ew))
            
