import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    if snap == '151':
        z = 0.
   
    Npixels = 8000
    vel_range = 600 # km/s 
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/data/samples/'
    spectra_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_los_rhoTZ_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_rho = np.zeros(len(gal_ids) * len(orients) * Npixels)
    all_T = np.zeros(len(gal_ids) * len(orients) * Npixels)
    all_Z = np.zeros(len(gal_ids) * len(orients) * Npixels)

    start = 0
    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')
            all_rho[start:start+Npixels] = spectrum['phys_density']
            all_T[start:start+Npixels] = spectrum['temperature']
            all_Z[start:start+Npixels] = spectrum['metallicity']
            start += Npixels
    
    with h5py.File(results_file, 'a') as hf:
        if not f'los_rho_{fr200}r200' in hf.keys():
            hf.create_dataset(f'los_rho_{fr200}r200', data=np.array(all_rho))
        if not f'los_T_{fr200}r200' in hf.keys():
            hf.create_dataset(f'los_T_{fr200}r200', data=np.array(all_T))
        if not f'los_Z_{fr200}r200' in hf.keys():
            hf.create_dataset(f'los_Z_{fr200}r200', data=np.array(all_Z))
