import pygad as pg
import h5py
import numpy as np
import caesar
import sys
from astropy.cosmology import FlatLambdaCDM
import astropy.io.ascii as ascii
from generate_spectra_pygad_trident import generate_trident_spectrum

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'

# load information about COS-Halos galaxies
cos_data = ascii.read('/home/sapple/cgm/cos_data.dat')
cos_b = cos_data['b(kpc)']
cos_M = cos_data['log10(M_*)(Msun)']

sim = caesar.load(infile)
ds = yt.load(snapfile)

h = sim.simulation.hubble_constant
hubble = h*1.e2
c = yt.units.c.in_units('km/s')

line_tri = 'H I 1216'
lambda_rest = 1215.6701

gals = sim.central_galaxies
stellar_masses = np.array([gals[i].masses['stellar'].in_units('Msun') for i in range(len(gals))])
stellar_masses = np.log10(stellar_masses)
positions = np.array([gals[i].pos.in_units('kpc/h') for i in range(len(gals))])
vels = np.array([gals[i].vel for i in range(len(gals))])

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
			los = np.array(pos_sample[j]); los[ax] = ds.domain_left_edge[ax]; los[rolled[ax]] += cos_b[i]
			vgal = vels_sample[j] + 1.e-3*hubble*pos_sample[j][2]
			generate_trident_spectrum(ds, line_tri, los, line_tri + 'cos_'+str(i)+'_gal_'+str(j)+'_x_plus', lambda_rest)

			los = np.array(pos_sample[j]); los[ax] = ds.domain_left_edge[ax]; los[rolled[ax]] -= cos_b[i]
               	 	vgal = vels_sample[j] + 1.e-3*hubble*pos_sample[j][2]
                	generate_trident_spectrum(ds, line_tri, los, line_tri + 'cos_'+str(i)+'_gal_'+str(j)+'_x_minus', lambda_rest)     
