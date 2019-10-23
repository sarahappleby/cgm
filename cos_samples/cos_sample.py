import sys
import os
import numpy as np
import caesar
from pyigm.cgm import cos_halos as pch
import yt
import h5py
import matplotlib.pyplot as plt

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]

sample_dir = sys.argv[4]
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)

touch(sample_dir+ '/cos_galaxy_samples.h5')

mass_range = 0.125

cos_halos = pch.COSHalos()
cos_M = []
cos_ssfr = []
cos_rho = []
for cos in cos_halos:
	cos = cos.to_dict()
	cos_M.append(cos['galaxy']['stellar_mass'])
	cos_ssfr.append(cos['galaxy']['ssfr'])
	cos_rho.append(cos['rho'])
cos_rho = yt.YTArray(cos_rho, 'kpc')

plt.scatter(cos_M, np.log10(cos_ssfr), c='k', marker='x', label='COS-Halos')

print('Loaded COS-Halos survey data')

infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
sim = caesar.load(infile, LoadHalo=False)
gal_cent = np.array([i.central for i in sim.galaxies])

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')

gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession

print('Loaded caesar galaxy data from model ' + model + ' snapshot ' + snap)

choose_mask = np.ones(len(gal_sm))

gal_ids = np.zeros(44*5)
mass = np.zeros(44*5)
ssfr = np.zeros(44*5)
pos = np.zeros((44*5, 3))
vgal_pos = np.zeros((44*5, 3))

for cos_id in range(44):
        ids = range(cos_id*5, (cos_id+1)*5)
	print('\nFinding the caesar galaxies in the mass and ssfr range of COS Halos galaxy ' + str(cos_id))
	
	mass_mask = (gal_sm > (cos_M[cos_id] - mass_range)) & (gal_sm < (cos_M[cos_id] + mass_range))
	stop = False
	init = 0.1
	while not stop:
		ssfr_mask = (gal_ssfr > (1. - init)*cos_ssfr[cos_id]) & (gal_ssfr < (1. + init)*cos_ssfr[cos_id])
		mask = mass_mask * ssfr_mask * gal_cent * choose_mask
		indices = np.where(mask == True)[0]
		if len(indices) < 5.: 
			init += 0.1
			continue
		else:
			stop = True
			continue

	# choose 5 of the galaxies that satisfy the COS-Halos galaxy's conditions
	choose = np.sort(np.random.choice(range(len(indices)), 5, replace=False))
	print('Chosen galaxies ' + str(indices[choose]))
	gal_ids[ids] = indices[choose]
        mass[ids] = gal_sm[indices[choose]]
	ssfr[ids] = gal_ssfr[indices[choose]]
	pos[ids] = gal_pos[indices[choose]]
	vgal_pos[ids] = gal_vgal_pos[indices[choose]]

	# do not repeat galaxies
        choose_mask[indices[choose]] = np.zeros(5)

with h5py.File(sample_dir+'/'+model+'_'+wind+'_cos_galaxy_sample.h5', 'a') as hf:
        hf.create_dataset('gal_ids', data=np.array(gal_ids))
        hf.create_dataset('mass', data=np.array(mass))
        hf.create_dataset('ssfr', data=np.array(ssfr))
        hf.create_dataset('position', data=np.array(pos))
        hf.create_dataset('vgal_position', data=np.array(vgal_pos))
        hf.attrs['pos_units'] = 'kpc/h'
        hf.attrs['mass_units'] = 'Msun'
        hf.attrs['ssfr_units'] = 'Msun/yr'
        hf.attrs['vel_units'] = 'km/s'

ssfr[np.where(ssfr == 0.)] = 1.e-14
plt.scatter(mass, np.log10(ssfr), s=2., c='b', label='Simba')
plt.xlabel('log M*')
plt.ylabel('log sSFR')
plt.ylim(-14.5, )
plt.legend()
plt.savefig(sample_dir + '/'+model+'_'+wind+'_sample_plot.png')
plt.clf()

