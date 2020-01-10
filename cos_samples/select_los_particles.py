from pygadgetreader import readsnap
import numpy as np
import h5py
import caesar
import gc

# for each galaxy in the sample, identify particles that contribute towards the line of sight using the smoothing length of each particle
# save new dataset containing only these particles using Chris's approach

model = 'm100n1024'
wind = 's50'
survey = 'dwarfs'
mlim = np.log10(5.8e8)

if survey == 'dwarfs':
    from get_cos_info import get_cos_dwarfs
    cos_rho, cos_M, cos_ssfr = get_cos_dwarfs()
    snap = '151'
elif survey == 'halos':
    from get_cos_info import get_cos_halos
    cos_rho, cos_M, cos_ssfr = get_cos_halos()
    snap = '137'

data_dir = '/home/rad/data/'+model+'/'+wind+'/'

sim = caesar.load(data_dir+'Groups/'+model+'_'+snap+'.hdf5', LoadHalo=False)
gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies]) # in kpc/h
h = sim.simulation.hubble_constant
redshift = sim.simulation.redshift

snapfile = data_dir + 'snap_'+model+'_'+snap +'.hdf5' 
# need PartType0 - SmoothingLength
hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h
gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h

sample_dir = '/home/sapple/cgm/cos_samples/cos_'+survey+'/samples/'
sample_file = sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5'
with h5py.File(sample_file, 'r') as f:
    gal_ids = f['gal_ids'][:].astype('int')
    gal_pos = f['position'][:] / (1.+redshift) # already in kpc/h, factor of 1+z for comoving

cos_rho = cos_rho[cos_M > mlim]
cos_rho = (np.repeat(cos_rho, 5) * h ) / (1+redshift)

keep_particles = np.array([False] * len(hsml))

for i, gal in enumerate(gal_ids):

    pos = gal_pos[i]
    los = np.array([pos[:2].copy(), ]*4)
    los[0][0] += cos_rho[i]
    los[1][0] -= cos_rho[i]
    los[2][1] += cos_rho[i]
    los[3][1] -= cos_rho[i]

    for l in los:
        x_dist = np.abs(l[0] - gas_pos[:, 0])
        y_dist = np.abs(l[1] - gas_pos[:, 1])
        dist_mask = (x_dist < hsml) & (y_dist < hsml)
        keep_particles[dist_mask] = True

        del x_dist, y_dist, dist_mask; gc.collect()

with h5py.File(sample_dir+'particles_needed.h5', 'a'):
    f.create_dataset('particle_mask', data=np.array(keep_particles))
