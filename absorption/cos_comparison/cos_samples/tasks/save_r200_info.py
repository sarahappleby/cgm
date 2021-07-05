import sys
import numpy as np
import caesar
import h5py
import sys

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

infile = f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5'
sim = caesar.load(infile)
halo_r200 = np.array([i.virial_quantities['r200c'].in_units('kpc/h') for i in sim.halos])

sample_dir = f'/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/{model}/'

with h5py.File(f'{sample_dir}{model}_{snap}_halo_r200_.h5', 'a') as f:
    f.create_dataset(wind+'_halo_r200', data=np.array(halo_r200))
