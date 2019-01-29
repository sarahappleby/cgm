import sys
import os
import numpy as np
import h5py

import yt
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

# need to get the impact parameters from the COS-Halos survey data:
cos_halos = pch.COSHalos()
cos_rho = []
for cos in cos_halos:
        cos = cos.to_dict()
        cos_rho.append(cos['rho'])

# Get some info from yt:
ds = yt.load(snapfile)

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)

# Set some spectrum parameters:
snr = 12.
periodic_vel = True
vel_range = 600. # km/s
sigma_vel = 6. # km/s
Nbins = int(np.rint(vbox / sigma_vel))

# Load in data for the sample galaxies corresponding to this COS-Halos galaxy
cos_sample = h5py.File(save_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5', 'r')
gal_ids = cos_sample['gal_ids'].value

# we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
# so instead we convert to default units of s['pos']
# hence kpc/h and the factor of (1+z) is necessary
pos_sample = cos_sample['position'].value*(1.+ds.current_redshift)
vgal_position_sample = cos_sample['vgal_position'].value
cos_sample.close()	

# Load in snapshot for pygad spectra generation:
s = pg.Snap(snapfile)

# Generate spectra for each line of sight:
for i in range(len(gal_ids)):
	print 'Generating spectra for sample galaxy ' + str(gal_ids[i])
	gal_name = 'cos_galaxy_'+str(cos_id)+'_sample_galaxy_' + str(gal_ids[i]) + '_'

	spec_name = gal_name + 'x_plus'
	los = pos_sample[i][:2].copy(); los[0] += cos_rho[cos_id]
	print 'In kpc/h: ' + str(los)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)
	
	spec_name = gal_name + 'x_minus'
	los = pos_sample[i][:2].copy(); los[0] -= cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + 'y_plus'
	los = pos_sample[i][:2].copy(); los[1] += cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)
	
	spec_name = gal_name + 'y_minus'	
	los = pos_sample[i][:2].copy(); los[1] -= cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)
