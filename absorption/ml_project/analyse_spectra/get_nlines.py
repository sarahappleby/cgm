import h5py
import numpy as np
import pygad as pg
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    #spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/no_uvb/{model}_{wind}_{snap}/'

    s = pg.Snapshot(f'{sample_dir}{model}_{wind}_{snap}.hdf5')
    redshift = s.redshift

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_n = np.zeros((len(gal_ids), len(orients)))

    #results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_nlines_{line}.h5'
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/{model}_{wind}_{snap}_no_uvb_nlines_{line}.h5'

    if os.path.isfile(results_file):
        with h5py.File(results_file, 'r') as f:
            all_keys = list(f.keys())
    else:
        all_keys = []

    if not f'ew_wave_{fr200}r200' in all_keys:

        for i in range(len(gal_ids)):
            for o, orient in enumerate(orients):
                spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
                spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

                if not 'line_list' in spectrum.keys():
                    spectrum['line_list'] = {}
                    spectrum['line_list']['N'] = []

                all_n[i][o] = len(spectrum['line_list']['N']) 
   
        with h5py.File(results_file, 'a') as hf:
            if not f'nlines_{fr200}r200' in hf.keys():
                hf.create_dataset(f'nlines_{fr200}r200', data=np.array(all_n))
