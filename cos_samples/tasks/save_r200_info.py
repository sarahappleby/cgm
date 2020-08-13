import sys
import numpy as np
import caesar
import h5py
import sys

model = sys.argv[1]
wind = sys.argv[2]
survey = sys.argv[3]
if survey == 'halos': snap = '137'
elif survey == 'dwarfs': snap = '151'

infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
sim = caesar.quick_load(infile)
halo_r200 = np.array([i.virial_quantities['r200c'].in_units('kpc/h') for i in sim.halos])

sample_dir = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'
with h5py.File(sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5', 'r') as cos_sample:
    gal_ids = cos_sample['gal_ids'][:]

r200_sample = np.zeros(len(gal_ids))
r200_sample[np.isnan(gal_ids)] = np.nan
r200_sample[~np.isnan(gal_ids)] = halo_r200[gal_ids[~np.isnan(gal_ids)].astype(np.int64)]

with h5py.File(sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5', 'a') as f:
    f.create_dataset('r200', data=np.array(r200_sample))
