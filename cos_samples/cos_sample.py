import sys
import os
import numpy as np
import caesar
from pyigm.cgm import cos_halos as pch
import yt

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]

sample_dir = '/home/sapple/cgm/cos_samples/pygad/periodic/kpch/samples/'

infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'

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
cos_rho = YTArray(cos_rho, 'kpc')

print 'Loaded COS-Halos survey data'

sim = caesar.load(infile)
gals = sim.central_galaxies

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')

stellar_masses = yt.YTArray([gals[i].masses['stellar'].in_units('Msun') for i in range(len(gals))], 'Msun')
sfr = np.array([gals[i].sfr.in_units('Msun/yr') for i in range(len(gals))])
ssfr = sfr / stellar_masses
positions = yt.YTArray([gals[i].pos.in_units('kpc/h') for i in range(len(gals))], 'kpc/h')
vels = yt.YTArray([gals[i].vel.in_units('km/s') for i in range(len(gals))], 'km/s')
stellar_masses = np.log10(stellar_masses)
recession = positions.in_units('kpc')*hubble
vgal_position = vels + recession

print 'Loaded caesar galaxy data from model ' + model + ' snapshot ' + snap

for cos_id in range(44):
	if os.path.isfile(sample_dir+'cos_galaxy_'+str(cos_id)+'_sample_data.h5'):
		continue
	else:

		print '\nFinding the caesar galaxies in the mass and ssfr range of COS Halos galaxy ' + str(cos_id)
		
		mass_mask = (stellar_masses > (cos_M[cos_id] - mass_range)) & (stellar_masses < (cos_M[cos_id] + mass_range))
		stop = False
		init = 0.1
		while not stop:
			ssfr_mask = (ssfr > (1. - init)*cos_ssfr[cos_id]) & (ssfr < (1. + init)*cos_ssfr[cos_id])
			mask = mass_mask * ssfr_mask
			indices = np.where(mask == True)[0]
			if len(indices) < 5.: 
				init += 0.1
				continue
			else:
				stop = True
				continue

		# choose 5 of the galaxies that satisfy the COS-Halos galaxy's conditions
		choose = np.sort(np.random.choice(range(len(indices)), 5, replace=False))
		print 'Chosen galaxies ' + str(indices[choose])
		gal_ids = indices[choose]

		mass_sample = stellar_masses[indices[choose]]
		ssfr_sample = ssfr[indices[choose]]
		pos_sample = positions[indices[choose]]
		vels_sample = vels[indices[choose]]
		vgal_position_sample = vgal_position[indices[choose]]

		del gals, stellar_masses, ssfr, mass_mask, ssfr_mask

		with h5py.File(sample_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5', 'w') as hf:
			hf.create_dataset('mask', data=np.array(mask))
			hf.create_dataset('gal_ids', data=np.array(indices[choose]))
			hf.create_dataset('mass', data=np.array(mass_sample))
			hf.create_dataset('ssfr', data=np.array(ssfr_sample))
			hf.create_dataset('position', data=np.array(pos_sample))
			hf.create_dataset('vgal_position', data=np.array(vgal_position_sample))
			hf.attrs['pos_units'] = 'kpc/h'
			hf.attrs['mass_units'] = 'Msun'
			hf.attrs['ssfr_units'] = 'Msun/yr'
			hf.attrs['vel_units'] = 'km/s'
		hf.close()
