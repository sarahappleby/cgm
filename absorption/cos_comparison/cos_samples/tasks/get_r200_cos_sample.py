import sys
import numpy as np
import caesar
import h5py

if __name__ == '__main__':

    survey = sys.argv[1]
    model = sys.argv[2]
    wind = sys.argv[3]
    
    if survey == 'dwarfs':
        snap = '151'
    elif survey == 'halos':
        snap = '137'

    infile = f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5'
    sim = caesar.load(infile)

    sample_file = f'/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/{model}/cos_{survey}/samples/{model}_{wind}_cos_{survey}_sample.h5'
    with h5py.File(sample_file, 'r') as f:
        gal_ids = np.array(f['gal_ids'][:])

    halo_r200_sample = np.ones(len(gal_ids)) * np.nan
    halo_mass_sample = np.ones(len(gal_ids)) * np.nan

    halo_r200_sample[~np.isnan(gal_ids)] = \
            np.array([sim.galaxies[int(i)].halo.virial_quantities['r200c'].in_units('kpc/h') for i in gal_ids if ~np.isnan(i)])
    halo_mass_sample[~np.isnan(gal_ids)] = \
            np.log10(np.array([sim.galaxies[int(i)].halo.masses['total'].in_units('Msun') for i in gal_ids if ~np.isnan(i)]))

    with h5py.File(sample_file, 'a') as f:
        del f['halo_r200']
        f.create_dataset('halo_r200', data=np.array(halo_r200_sample))
        f.create_dataset('halo_mass', data=np.array(halo_mass_sample))

