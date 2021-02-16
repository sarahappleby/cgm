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
from ignore_gals import *

mlim = np.log10(5.8e8)
sqrt2 = np.sqrt(2.)

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
survey = sys.argv[4]
background = sys.argv[5]
num = int(sys.argv[6])
line = sys.argv[7]
lambda_rest = float(re.findall(r'\d+', line)[0])
ngals_each = 5


ids = list(range(num*ngals_each, (num+1)*ngals_each))

if ((model == 'm50n512') & (survey == 'halos')) or (model == 'm25n512') or (model == 'm25n256'):
    ignore_cos_gals, ngals_each = get_ignore_cos_gals(model, survey)
    if num in ignore_cos_gals:
        print('Ignoring certain m50n512 COS-Halos galaxies')
        import sys
        sys.exit() 

snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
snapfile = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_'+snap+'.hdf5'

sample_dir = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'
save_dir = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/'+wind+'/'+background+'/'

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
	os.makedirs(save_dir + '/plots/')
	os.makedirs(save_dir + '/spectra/')
	os.makedirs(save_dir + '/output/')

if survey == 'dwarfs':
	from get_cos_info import get_cos_dwarfs
	cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()

elif survey == 'halos':
	from get_cos_info import get_cos_halos
	cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_halos()

# Get some info from yt:
ds = yt.load(snapfile)
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)

# Set some spectrum parameters:
snr = 12.
vel_range = 600. # km/s
pixel_size = 6 # km/s
periodic_vel = True
if periodic_vel:
	v_limits = [-600., vbox.value+600.]
else:
	v_limits = [0., vbox]
total_v = np.sum(np.abs(v_limits))
Nbins = int(np.rint(total_v / pixel_size))

# Load in data for the sample galaxies corresponding to this COS-Halos galaxy
with h5py.File(sample_dir+model+'_'+wind+'_cos_'+survey+'_sample.h5', 'r') as cos_sample:
	
	gal_ids = (cos_sample['gal_ids'][:])[ids]
	cos_ids = cos_sample['cos_ids'][:]
	# we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
	# so instead we convert to default units of s['pos']
	# hence kpc/h and the factor of (1+z) is necessary
	pos_sample = (cos_sample['position'][:]  *  (1.+ds.current_redshift))[ids]
	vgal_position_sample = (cos_sample['vgal_position'][:])[ids][:, 2]

cos_id = cos_ids[num]
cos_rho = cos_rho * (ds.hubble_constant * (1 + ds.current_redshift)) # originally in kpc, need in kpc/h to match pygad


# Load in snapshot for pygad spectra generation:
s = pg.Snapshot(snapfile)

# Generate spectra for each line of sight:
for i in list(range(len(gal_ids))):
	print('Generating spectra for sample galaxy ' + str(gal_ids[i]))
	gal_name = 'sample_galaxy_' + str(gal_ids[i]) + '_'

	spec_name = gal_name + '0_deg'
	los = pos_sample[i][:2].copy(); los[0] += cos_rho[cos_id]
	print('In kpc/h: ' + str(los))
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '45_deg'
	los = pos_sample[i][:2].copy(); los[0] += (cos_rho[cos_id] / sqrt2); los[1] += (cos_rho[cos_id] / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '90_deg'
	los = pos_sample[i][:2].copy(); los[1] += cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '135_deg'
	los = pos_sample[i][:2].copy(); los[0] -= (cos_rho[cos_id] / sqrt2); los[1] += (cos_rho[cos_id] / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '180_deg'
	los = pos_sample[i][:2].copy(); los[0] -= cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '225_deg'
	los = pos_sample[i][:2].copy(); los[0] -= (cos_rho[cos_id] / sqrt2); los[1] -= (cos_rho[cos_id] / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '270_deg'    
	los = pos_sample[i][:2].copy(); los[1] -= cos_rho[cos_id]
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

	spec_name = gal_name + '315_deg'
	los = pos_sample[i][:2].copy(); los[0] += (cos_rho[cos_id] / sqrt2); los[1] -= (cos_rho[cos_id] / sqrt2)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, v_limits, Nbins, snr, vgal_position_sample[i], vel_range, spec_name, save_dir)

