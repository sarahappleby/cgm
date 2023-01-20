# Make plots of the spectrum fits for LOS within a range of chisq values to compare the quality of the fits.

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

np.random.seed(0)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


def make_chisq_range_plots(chisq, min_chisq, max_chisq, gal_ids, line, fr200, orients, spectra_dir, plot_dir, ngals=10):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    chisq_index = np.where((chisq > min_chisq) & (chisq < max_chisq))
    
    success = False
    while not success:
        try:
            choice = np.random.choice(np.arange(len(chisq_index[0])), size=ngals, replace=False)
            success = True
        except ValueError:
            ngals -= 1
           
    if len(choice) == 0:
        print('No galaxies matching this criteria')
        return
    
    else:
        first, second = chisq_index[0][choice], chisq_index[1][choice]
        for i in range(ngals):
            spec_name = f'sample_galaxy_{gal_ids[first[i]]}_{line}_{orients[second[i]]}_{fr200}r200'
            spec = Spectrum(f'{spectra_dir}{spec_name}.h5')
            spec.plot_fit(filename = f'{plot_dir}/{spec_name}.png')
    
        return

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    ngals = 10
    orients = np.array(['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'])

    spectra_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    sample_dir = f'/disk04/sapple/data/samples/'
    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/chisq_plots/{line}_{fr200}r200/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    results_dir = f'/disk04/sapple/data/normal/results/'
    chisq_file = f'{results_dir}{model}_{wind}_{snap}_fit_chisq_{line}.h5'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    chisq_dict = read_h5_into_dict(chisq_file)
    max_chisq = chisq_dict[f'max_chisq_{fr200}r200'] 

    #make_chisq_range_plots(max_chisq, 20, 500, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_min_20/')
    make_chisq_range_plots(max_chisq, 30, 50, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_30-50/')

    #make_chisq_range_plots(max_chisq, 2.9, 3.1, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_2.9_3.1/')
    #make_chisq_range_plots(max_chisq, 3.9, 4.1, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_3.9_4.1/')
    #make_chisq_range_plots(max_chisq, 4.9, 5.1, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_4.9_5.1/')

    #make_chisq_range_plots(max_chisq, 0., 2.5, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_0_2.5/')
    #make_chisq_range_plots(max_chisq, 2.5, 5., gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_2.5_5/')
    #make_chisq_range_plots(max_chisq, 5., 10., gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_5_10/')
    #make_chisq_range_plots(max_chisq, 10., 100., gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/chisq_10_100/')
    #make_chisq_range_plots(max_chisq, -99.5, -89.5, gal_ids, line, fr200, orients, spectra_dir, f'{plot_dir}/no_features/')
