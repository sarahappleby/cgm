import sys
sys.path.append('/home/sapple/VAMP')

from vpfits import *

import h5py

cos_dir = '/home/sapple/cgm/cos_samples/pygad/'
sample_dir = cos_dir + 'samples/'
output_dir = '/home/sapple/cgm/fitting/'

icos = 0
line = 'H1215'
lambda_rest = float(filter(str.isdigit, line))

cos_samples = h5py.File(sample_dir+'cos_galaxy_'+str(icos)+'_sample_data.h5')
gal_ids = cos_samples['gal_ids'].value
vgal_position = cos_samples['vgal_position'].value

directions = ['x_plus', 'x_minus', 'y_plus', 'y_minus']

for igal in range(len(gal_ids)):
	for d in directions:
		spectrum_file = cos_dir + 'spectra/cos_galaxy_'+ str(icos)+'_sample_galaxy_'+str(igal)+'_'+d + '.h5'
		with h5py.File(spectrum_file, 'r') as spectrum:
			wavelength = spectrum['wavelength'].value
			velocity = spectrum['velocity'].value
			flux = spectrum[line+'_flux'].value
			noise = spectrum['noise'].value
			tau = spectrum[line+'_tau'].value
		fit_file = 'cos_galaxy_'+ str(icos)+'_sample_galaxy_'+str(igal)+'_'+d+'_'
		params, flux_model = fit_spectrum(wavelength, noise, tau, lambda_rest, voigt=False, folder=output_dir+fit_file)
		write_ascii(params, output_dir+fit_file+'params.dat')
