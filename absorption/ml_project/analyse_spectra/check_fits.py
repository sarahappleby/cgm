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


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


def make_chisq_range_plots(chisq, min_chisq, max_chisq, gal_ids, line, fr200, orients, spectra_dir, plot_dir, ngals=10):

    chisq_index = np.where((chisq > min_chisq) & (chisq < max_chisq))
    choice = np.random.choice(np.arange(len(chisq_index[0])), size=ngals, replace=False)
    first, second = chisq_index[0][choice], chisq_index[1][choice]
    for i in range(ngals):
        spec_name = f'sample_galaxy_{gal_ids[first[i]]}_{line}_{orients[second[i]]}_{fr200}r200'
        spec = Spectrum(f'{spectra_dir}{spec_name}.h5')
        spec.plot_fit(filename = f'{plot_dir}/{spec_name}.png')


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

    make_chisq_range_plots(max_chisq, 0., 2.5, gal_ids, line, fr200, orients, spectra_dir, 'chisq_plots/low_chisq/')
    make_chisq_range_plots(max_chisq, 2.5, 5., gal_ids, line, fr200, orients, spectra_dir, 'chisq_plots/mid_chisq/')
    make_chisq_range_plots(max_chisq, 5., np.inf, gal_ids, line, fr200, orients, spectra_dir, 'chisq_plots/high_chisq/')
    make_chisq_range_plots(max_chisq, -99.5, -89.5, gal_ids, line, fr200, orients, spectra_dir, 'chisq_plots/no_features/')

    plt.hist(np.log10(chisq), bins=100, density=True, alpha=0.6)
    plt.axvline(np.log10(2.5), ls='--', c='k')
    plt.xlabel(r'${\rm log}\ \chi^2_r$')
    plt.ylabel(r'${\rm Frequency}$')
    plt.savefig(f'chisq_plots/chisq_hist_{line}_{fr200}r200.png')
    plt.clf()

