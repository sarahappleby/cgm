# Script to identify LOS particles for a given selected galaxy, for LOS parallel to the z axis of the simulation
# Run using sub_los_particles.sh

from pygadgetreader import readsnap
import numpy as np
import h5py
import caesar
import gc
import sys
from ignore_gals import *

# for each galaxy in the sample, identify particles that contribute towards the line of sight using the smoothing length of each particle
# save new dataset containing only these particles using Chris's approach

sqrt2 = np.sqrt(2.)
model = sys.argv[1]
wind = sys.argv[2]
sample_gal = int(sys.argv[3]) # supply the gal id that we want from command line
survey = sys.argv[4]
mlim = np.log10(5.8e8)
ngals_each = 5

# new: for the m50n512 samples, ignore certain COS-Halos galaxies for which there are insufficient Simba analogs.
# Also, for certain COS-Dwarf galaxies in the m25n512 sample.
if ((model == 'm50n512') & (survey == 'halos')) or (model == 'm25n512') or (model == 'm25n256'):
    ignore_simba_gals, ngals_each = get_ignore_simba_gals(model, survey)
    if sample_gal in ignore_simba_gals:
        print('Ignoring certain m50n512 COS-Halos galaxies')
        import sys
        sys.exit()

if survey == 'dwarfs':
    from get_cos_info import get_cos_dwarfs
    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()
    snap = '151'
elif survey == 'halos':
    from get_cos_info import get_cos_halos
    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_halos()
    snap = '137'

data_dir = '/home/rad/data/'+model+'/'+wind+'/'

sim = caesar.quick_load(data_dir+'Groups/'+model+'_'+snap+'.hdf5')
gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies]) # in kpc/h
h = sim.simulation.hubble_constant
redshift = sim.simulation.redshift

snapfile = data_dir + 'snap_'+model+'_'+snap +'.hdf5' 
# need PartType0 - SmoothingLength
hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving

sample_dir = '/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/'+model+'/cos_'+survey+'/samples/'
sample_file = sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5'
with h5py.File(sample_file, 'r') as f:
    gal_id = f['gal_ids'][:].astype('int')[sample_gal]
    pos = f['position'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving

cos_rho = cos_rho[cos_M > mlim]
cos_rho = (np.repeat(cos_rho, ngals_each) * h ) * (1+redshift)

los = np.array([pos[:2].copy(), ]*8)
los[0][0] += cos_rho[sample_gal]
los[1][0] += (cos_rho[sample_gal] / sqrt2); los[1][1] += (cos_rho[sample_gal] / sqrt2)
los[2][1] += cos_rho[sample_gal]
los[3][0] -= (cos_rho[sample_gal] / sqrt2); los[3][1] += (cos_rho[sample_gal] / sqrt2)
los[4][0] -= cos_rho[sample_gal]
los[5][0] -= (cos_rho[sample_gal] / sqrt2); los[5][1] -= (cos_rho[sample_gal] / sqrt2)
los[6][1] -= cos_rho[sample_gal]
los[7][0] += (cos_rho[sample_gal] / sqrt2); los[7][1] -= (cos_rho[sample_gal] / sqrt2)

partids = np.array([])

for l in los:
    x_dist = np.abs(l[0] - gas_pos[:, 0])
    y_dist = np.abs(l[1] - gas_pos[:, 1])
    hyp_sq = x_dist**2 + y_dist**2
    dist_mask = hyp_sq < hsml**2
    #dist_mask = (x_dist < hsml) & (y_dist < hsml)
    partids = np.append(partids, np.arange(len(hsml))[dist_mask])
    
    del x_dist, y_dist, dist_mask; gc.collect()

partids = np.unique(np.sort(partids))

with h5py.File(sample_dir+wind+'_particle_selection.h5', 'a') as f:
    f.create_dataset('plist_'+str(sample_gal)+'_'+str(gal_id), data=np.array(partids))

del partids
