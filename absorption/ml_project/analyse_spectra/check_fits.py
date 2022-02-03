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


    if not os.path.isfile(chisq_file):

        all_max_chisq = np.zeros((len(gal_ids), len(orients)))
    
        for i in range(len(gal_ids)):
            for o, orient in enumerate(orients):
                spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
                spectrum = read_h5_into_dict(f'{spectra_dir}{spec_name}.h5')

                if len(spectrum['lines']['fit_Chisq']) > 0.:
                    all_max_chisq[i][o] = np.nanmax(spectrum['lines']['fit_Chisq'])
                else:
                    all_max_chisq[i][o] = -9999.

        with h5py.File(chisq_file, 'a') as hf:
            if not f'max_chisq_{fr200}r200' in hf.keys():
                hf.create_dataset(f'max_chisq_{fr200}r200', data=np.array(all_max_chisq))

    max_chisq_dict = read_h5_into_dict(chisq_file)
    max_chisq = max_chisq_dict[f'max_chisq_{fr200}r200']

    low_chisq = np.where((max_chisq > 0.) & (max_chisq < 2.5))
    choice = np.random.choice(np.arange(len(low_chisq[0])), size=ngals, replace=False)
    first, second = low_chisq[0][choice], low_chisq[1][choice]
    gals = gal_ids[first]

    for i in range(ngals):
        spec_name = f'sample_galaxy_{gal_ids[first[i]]}_{line}_{orients[second[i]]}_{fr200}r200'
        spec = Spectrum(f'{spectra_dir}{spec_name}')
    
    mid_chisq = np.where((max_chisq > 2.5) & (max_chisq < 5.))
    high_chisq = np.where((max_chisq > 5.))
    no_features = np.where((max_chisq == -9999.))


