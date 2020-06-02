from pygadgetreader import readsnap
import numpy as np
import h5py
import caesar
import gc
import sys

# for each galaxy in the sample, identify particles that contribute towards the line of sight using the smoothing length of each particle
# save new dataset containing only these particles using Chris's approach

# SA: modify this such that we are running for one galaxy at a time to save some memory potentially
# Also, check it works by running new spectrum with saved data

# SA: write python wrapper type thing for this that runs it with every gal id from the sample
# SA: write bash script to submit the cos id and the number 0-4 for the galaxy, run python wrapper which then looks up the gal id and supplies it to this script

sample_gal = int(sys.argv[1]) # supply the gal id that we want from command line
survey = sys.argv[2]
model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8)

if survey == 'dwarfs':
    from get_cos_info import get_cos_dwarfs
    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()
    snap = '151'
elif survey == 'halos':
    from get_cos_info import get_cos_halos
    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_halos()
    snap = '137'

data_dir = '/home/rad/data/'+model+'/'+wind+'/'

sim = caesar.load(data_dir+'Groups/'+model+'_'+snap+'.hdf5', LoadHalo=False)
gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies]) # in kpc/h
h = sim.simulation.hubble_constant
redshift = sim.simulation.redshift

snapfile = data_dir + 'snap_'+model+'_'+snap +'.hdf5' 
# need PartType0 - SmoothingLength
hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving

sample_dir = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'
sample_file = sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5'
with h5py.File(sample_file, 'r') as f:
    gal_id = f['gal_ids'][:].astype('int')[sample_gal]
    pos = f['position'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving

cos_rho = cos_rho[cos_M > mlim]
cos_rho = (np.repeat(cos_rho, 5) * h ) * (1+redshift)

los = np.array([pos[:2].copy(), ]*4)
los[0][0] += cos_rho[sample_gal]
los[1][0] -= cos_rho[sample_gal]
los[2][1] += cos_rho[sample_gal]
los[3][1] -= cos_rho[sample_gal]

partids = np.array([])

for l in los:
    x_dist = np.abs(l[0] - gas_pos[:, 0])
    y_dist = np.abs(l[1] - gas_pos[:, 1])
    dist_mask = (x_dist < hsml) & (y_dist < hsml)
    partids = np.append(partids, np.arange(len(hsml))[dist_mask])
    
    del x_dist, y_dist, dist_mask; gc.collect()

partids = np.unique(np.sort(partids))

with h5py.File(sample_dir+'particle_selection.h5', 'a') as f:
    f.create_dataset('plist_'+str(sample_gal)+'_'+str(gal_id), data=np.array(partids))

del partids; gc.collect()
