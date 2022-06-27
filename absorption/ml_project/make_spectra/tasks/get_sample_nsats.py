# Saving out the number of satellites each sample galaxy has :) 

import numpy as np
import h5py
import sys
import caesar

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
 
    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

    nsats = np.array([len(i.satellites) for i in sim.galaxies])[gal_ids]

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as sf:
        sf.create_dataset('nsats', data=np.array(nsats))
