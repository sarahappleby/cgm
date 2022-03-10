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
                all_chisq.extend(spectrum['line_list']['Chisq'])
                all_N.extend(spectrum['line_list']['N'])
                all_b.extend(spectrum['line_list']['b'])
                all_l.extend(spectrum['line_list']['l'])
                all_ew.extend(spectrum['line_list']['EW'])
                all_ids.extend([gal_ids[i]] * len(spectrum['line_list']['N']))
                            
    
    with h5py.File(results_file, 'a') as hf:
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
