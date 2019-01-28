import sys
import os
import numpy as np
import h5py

import yt
import caesar
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg
from pyigm.cgm import cos_halos as pch

from generate_spectra import generate_pygad_spectrum

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
cos_id = int(sys.argv[4])
line = sys.argv[5]
lambda_rest = float(filter(str.isdigit, line))
snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
save_dir = '/home/sapple/cgm/cos_samples/pygad/periodic/kpch/'

save_dir = '/home/sapple/cgm/cos_samples/test/'

cos_halos = pch.COSHalos()
cos_rho = []
for cos in cos_halos:
        cos = cos.to_dict()
        cos_rho.append(cos['rho'])
cos_rho = pg.UnitArr(cos_rho, 'kpc')

ds = yt.load(snapfile)
s = pg.Snap(snapfile)

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)
c = pg.physics.c.in_units_of('km/s')
snr = 12.
periodic_vel = True

sigma_vel = 6.
Nbins = int(np.rint(vbox / sigma_vel))


cos_sample = h5py.File(save_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5', 'r')
gal_ids = cos_sample['gal_ids'].value
# we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
# so instead we convert to default units of s['pos']
# hence kpc/h and the factor of (1+z) is necessary
pos_sample = cos_sample['position'].value*(1.+s.redshift)
vgal_position_sample = cos_sample['vgal_position'].value
cos_sample.close()	

for i in range(len(gal_ids)):
	print 'Generating spectra for sample galaxy ' + str(gal_ids[i])
	gal_name = 'cos_galaxy_'+str(cos_id)+'_sample_galaxy_' + str(gal_ids[i]) + '_'

	# for pygad:

	# pygad assumes same units as s['pos'], so need to convert from kpc/h to ckpc/h_0	
	print 'Converting galaxy positions from kpc/h to ckpc/h_0 for pygad'

	spec_name = gal_name + 'x_plus'
	los = pos_sample[i][:2].copy(); los[0] += cos_rho[cos_id]
	print 'In kpc/h: ' + str(los)
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')	
	#print 'In ckpc/h_0: ' + str(los)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], c, spec_name, save_dir)
	
	spec_name = gal_name + 'x_minus'
	los = pos_sample[i][:2].copy(); los[0] -= cos_rho[cos_id]
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], c, spec_name, save_dir)

	spec_name = gal_name + 'y_plus'
	los = pos_sample[i][:2].copy(); los[1] += cos_rho[cos_id]
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], c, spec_name, save_dir)
	
	spec_name = gal_name + 'y_minus'	
	los = pos_sample[i][:2].copy(); los[1] -= cos_rho[cos_id]
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], c, spec_name, save_dir)
