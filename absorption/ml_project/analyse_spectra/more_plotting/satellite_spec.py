import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'

    line = 'H1215'
    gal_id = 3301
    orient = '0_deg'
    log_frad = '1.0'
    fr200 = 0.25
    vel_range = 600.

    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf: 
        gal_ids = sf['gal_ids'][:]

    norm_spectra_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/' 
    sat_spectra_dir = f'/disk04/sapple/data/satellites/{model}_{wind}_{snap}/log_frad_{log_frad}/'

    spec_name = f'sample_galaxy_{gal_id}_{line}_{orient}_{fr200}r200'
    sat_spec = read_h5_into_dict(f'{sat_spectra_dir}{spec_name}.h5')
    norm_spec = read_h5_into_dict(f'{norm_spectra_dir}{spec_name}.h5')


    plt.plot(sat_spec['velocities'], sat_spec['fluxes'], label='Satellite only')
    plt.plot(norm_spec['velocities'], norm_spec['fluxes'], label='All gas')
    plt.xlim(sat_spec['gal_velocity_pos'] - vel_range, sat_spec['gal_velocity_pos'] + vel_range)
