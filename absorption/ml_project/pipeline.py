# Our main pipeline script to run pygad on each LOS
# Run using sub_line_pipeline.sh and sub_pipeline.sh

import sys
import os
import numpy as np
import h5py
import re
import yt
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg
from astropy.io import ascii
from generate_spectra import generate_pygad_spectrum

# Set some spectrum parameters:
sqrt2 = np.sqrt(2.)
snr = 12.
vel_range = 600. # km/s
pixel_size = 2.5 # km/s
periodic_vel = True
ngals_each = 12
delta_fr200 = 0.25
min_fr200 = 0.25
nbins_fr200 = 5
fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
num = int(sys.argv[4])
line = sys.argv[5]
lambda_rest = float(re.findall(r'\d+', line)[0])
ids = list(range(num*ngals_each, (num+1)*ngals_each))

snapfile = f'/disk01/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
s = pg.Snapshot(snapfile)

sample_dir = f'/disk01/sapple/cgm/absorption/ml_project/data/samples/'
save_dir = f'/disk01/sapple/cgm/absorption/ml_project/data/{model}_{wind}_{snap}/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

ds = yt.load(snapfile)
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)
if periodic_vel:
	v_limits = [-1.*vel_range, vbox.value+vel_range]
else:
	v_limits = [0., vbox]
total_v = np.sum(np.abs(v_limits))
Nbins = int(np.rint(total_v / pixel_size))

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
	
	gal_ids = sf['gal_ids'][:][ids]
	# we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
	# so instead we convert to default units of s['pos']
	# hence kpc/h and the factor of (1+z) is necessary
	pos_sample = (sf['position'][:]  *  (1.+ds.current_redshift))[ids]
	vgal_position_sample = sf['vgal_position'][:][ids][:, 2]
        r200 = sf['halo_r200'][:][ids] * (1.+ds.current_redshift) # already in kpc/h, factor of 1+z for comoving

for i in list(range(len(gal_ids))):

    for j in range(len(fr200)):

        rho = r200[i] * fr200[j]

	print('Generating spectra for sample galaxy ' + str(gal_ids[i]))
	gal_name = 'sample_galaxy_' + str(gal_ids[i]) + '_'

	spec_name = gal_name + '0_deg'
	los = pos_sample[i][:2].copy(); los[0] += rho
	print('In kpc/h: ' + str(los))
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '45_deg'
	los = pos_sample[i][:2].copy(); los[0] += (rho / sqrt2); los[1] += (rho / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '90_deg'
	los = pos_sample[i][:2].copy(); los[1] += rho
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '135_deg'
	los = pos_sample[i][:2].copy(); los[0] -= (rho / sqrt2); los[1] += (rho / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '180_deg'
	los = pos_sample[i][:2].copy(); los[0] -= rho
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '225_deg'
	los = pos_sample[i][:2].copy(); los[0] -= (rho / sqrt2); los[1] -= (rho / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '270_deg'    
	los = pos_sample[i][:2].copy(); los[1] -= rho
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

	spec_name = gal_name + '315_deg'
	los = pos_sample[i][:2].copy(); los[0] += (rho / sqrt2); los[1] -= (rho / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, spec_name, save_dir)

