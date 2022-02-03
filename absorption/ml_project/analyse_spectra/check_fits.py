import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from spectrum import Spectrum
from utils import *
from physics import *

if __name__ == '__main__':

    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    ngals = 10
    orients = np.array(['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'])

    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    chisq_file = f'{results_dir}fit_max_chisq_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    chisq_dict = read_h5_into_dict(chisq_file)
    max_chisq = chisq_dict[f'max_chisq_{fr200}r200'] 

    low_chisq = np.where((max_chisq > 0.) & (max_chisq < 2.5))
    choice = np.random.choice(np.arange(len(low_chisq[0])), size=ngals, replace=False)
    first, second = low_chisq[0][choice], low_chisq[1][choice]

    for i in range(ngals):
        spec_name = f'sample_galaxy_{gal_ids[first[i]]}_{line}_{orients[second[i]]}_{fr200}r200.h5'
        spec = Spectrum(f'{spectra_dir}{spec_name}')
        spec.plot_fit(f'plots/{spec_name}.png')
    
    mid_chisq = np.where((max_chisq > 2.5) & (max_chisq < 5.))
    high_chisq = np.where((max_chisq > 5.))
    no_features = np.where((max_chisq == -99.))

     
