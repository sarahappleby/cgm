# Forgot to save out the Mhalo of the galaxy sample :)

import numpy as np
import h5py
import sys
import caesar

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
 
    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

    mhalo_sample = np.log10([sim.galaxies[int(i)].halo.masses['total'].in_units('Msun') for i in gal_ids])


    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as sf:
        sf.create_dataset('halo_mass', data=np.array(mhalo_sample))
