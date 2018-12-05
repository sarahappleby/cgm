import pygad as pg
import h5py
import numpy as np
import caesar
import sys
from astropy.cosmology import FlatLambdaCDM
import astropy.io.ascii as ascii
from generate_spectra_pygad_trident import generate_trident_spectrum
from yt.units.yt_array import YTArray

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'

# load information about COS-Halos galaxies
cos_data = ascii.read('/home/sapple/cgm/cos_data.dat')
cos_b = YTArray(cos_data['b(kpc)'], 'kpc')
cos_M = np.array(cos_data['log10(M_*)(Msun)'])

sim = caesar.load(infile)
ds = yt.load(snapfile)

h = sim.simulation.hubble_constant
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_constant.in_units('km/s/kpc')
c = yt.units.c.in_units('km/s')

line_tri = 'H I 1216'
lambda_rest = 1215.6701

gals = sim.central_galaxies
stellar_masses = YTArray([gals[i].masses['stellar'].in_units('Msun') for i in range(len(gals))], 'Msun')
stellar_masses = np.log10(stellar_masses)
positions = YTArray([gals[i].pos.in_units('kpc/h') for i in range(len(gals))], 'kpc/h')
vels = YTArray([gals[i].vel.in_units('km/s') for i in range(len(gals))], 'km/s')

# example used in Ford et al 2016
# COS-halos galaxy with M=10**10.2 Msun and b = 18.26 kpc = 26.85 kpc/h
#cos_mass = 10.2
#cos_b = 26.85

mass_range = 0.125
snr = 12.
Nbins = 1000.

for i in range(len(cos_M)):

	mask = (stellar_masses > (cos_M[i] - mass_range)) & (stellar_masses < (cos_M[i] + mass_range)) 
	mass_sample = stellar_masses[mask]
	pos_sample = positions[mask]
	vels_sample = vels[mask]

	for j in range(len(mass_sample)):
				
		rolled = np.roll(range(3), -1)
		for ax in range(3): 
			ray_start = pos_sample[j].copy(); ray_start[ax] = ds.domain_left_edge[ax]; ray_start[rolled[ax]] += cos_b[i]
			ray_end = pos_sample[j].copy(); ray_end[ax] = ds.domain_right_edge[ax]; ray_end[rolled[ax]] += cos_b[i]
			vgal = vels_sample[j] + hubble*pos_sample[j][2]
			generate_trident_spectrum(ds, line_tri, ray_start, ray_end, line_tri + 'cos_'+str(i)+'_gal_'+str(j)+'_x_plus', lambda_rest)

			ray_start = pos_sample[j].copy(); ray_start[ax] = ds.domain_left_edge[ax]; ray_start[rolled[ax]] -= cos_b[i]
			ray_end = pos_sample[j].copy(); ray_end[ax] = ds.domain_right_edge[ax]; ray_end[rolled[ax]] -= cos_b[i]
               	 	vgal = vels_sample[j] + hubble*pos_sample[j][2]
                	generate_trident_spectrum(ds, line_tri, ray_start, ray_end, line_tri + 'cos_'+str(i)+'_gal_'+str(j)+'_x_minus', lambda_rest)     
