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

    vel_range = 600 # km/s 
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    ew_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_ew_{line}.h5'
    chisq_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_chisq_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_fit_ew = np.zeros((len(gal_ids), len(orients)))
    all_max_chisq = np.zeros((len(gal_ids), len(orients)))

    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

            if not 'line_list' in spectrum.keys():
                spectrum['line_list'] = {}
                spectrum['line_list']['N'] = []

            if len(spectrum['line_list']['N']) > 0.:
                all_fit_ew[i][o] = np.sum(spectrum['line_list']['EW'])
                all_max_chisq[i][o] = np.nanmax(spectrum['line_list']['Chisq'])
            
            else:
                all_max_chisq[i][o] = -99.

    with h5py.File(ew_file, 'a') as hf:
        if not f'fit_ew_{fr200}r200' in hf.keys():
            hf.create_dataset(f'fit_ew_{fr200}r200', data=np.array(all_fit_ew))

    with h5py.File(chisq_file, 'a') as hf:
        if not f'max_chisq_{fr200}r200' in hf.keys():
            hf.create_dataset(f'max_chisq_{fr200}r200', data=np.array(all_max_chisq))

