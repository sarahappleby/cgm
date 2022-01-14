import h5py
import numpy as np
import pygad as pg
import sys
from generate_spectra import read_spectrum

if __name__ == '__main__':
    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    n90_restr_column = np.zeros(len(gal_ids))

    for i in range(len(gal_ids)):
        n90_rc = np.zeros(len(orients))
        for o, orient in enumerate(orients):
            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spectrum = read_spectrum(f'{spectra_dir}{spec_name}.h5')
            n90_rc[o] = spectrum['n90_restr_column']

        n90_restr_column[i] = np.nanmedian(n90_rc)

    with h5py.File(f'data/normal/results/n90_restr_column_{line}.h5', 'a') as hf:
        hf.create_dataset(f'{fr200}r200', data=np.array(n90_restr_column))

