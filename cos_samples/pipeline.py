import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import h5py
import sys
import numpy as np

import yt
import trident
import caesar
from yt.units.yt_array import YTArray, YTQuantity

import pyigm
from pyigm.cgm import cos_halos as pch

import gc

import time
TINIT = time.time()
def t_elapsed(): return np.round(time.time()-TINIT,2)

def vel_to_wave(vel, lambda_rest, c, z):
	return lambda_rest * (1.0 + z) * (vel / c + 1.)

def wave_to_vel(wave, lambda_rest, c, z):
	return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)

def generate_trident_spectrum(ds, line_name, ray_start, ray_end, spec_name, lambda_rest, vpos, Nbins):
    print("Generating trident spectrum...")
    # Generate ray through box
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=[line_name],
                                  ftype='PartType0')

    #sg = trident.SpectrumGenerator(lambda_min=lambda_min.value, lambda_max=lambda_max, n_lambda=Nbins)
    sg = trident.SpectrumGenerator('COS-G130M')  # convolves with COS line spread fcn, gives COS resolution
    sg.make_spectrum(ray, lines=[line_name])

    # Get fields and convert wavelengths to velocities. Note that the velocities are negative compared to pygad!
    wavelengths = np.array(sg.lambda_field)
    taus = np.array(sg.tau_field)
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    fluxes = np.array(sg.flux_field)
    velocities = wave_to_vel(wavelengths, lambda_rest, c, ds.current_redshift)
    print 'Trident generated spectrum v=',min(velocities),max(velocities),len(wavelengths),len(velocities)

    plt.plot(velocities, fluxes)
    plt.axvline(vpos, linewidth=1, c='k')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux')
    plt.savefig('./plots/'+spec_name+'.png')
    plt.clf()

    #Save spectrum to hdf5 format
    with h5py.File('./spectra/{}.h5'.format(spec_name), 'w') as hf:
        hf.create_dataset('velocity', data=np.array(velocities))
        hf.create_dataset('flux', data=np.array(fluxes))
        hf.create_dataset('wavelength', data=np.array(wavelengths))
        hf.create_dataset('tau', data=np.array(taus))
        hf.create_dataset('noise', data=np.array(noise))
    print('Trident spectrum done [t=%g s]'%(t_elapsed()))
    
    del ray; gc.collect()
    return


model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
igal = int(sys.argv[4])
snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'

"""
# load information about COS-Halos galaxies
cos_data = ascii.read('/home/sapple/cgm/cos_data.dat')
cos_b = YTArray(cos_data['b(kpc)'], 'kpc')
cos_M = np.array(cos_data['log10(M_*)(Msun)'])
"""

cos_halos = pch.COSHalos()
cos_M = []
cos_rho = []
for sys in cos_halos:
	sys = sys.to_dict()
	cos_M.append(sys['galaxy']['stellar_mass'])
	cos_rho.append(sys['rho'])
cos_rho = YTArray(cos_rho, 'kpc')

sim = caesar.load(infile)
ds = yt.load(snapfile)

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)
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

vel_range = YTQuantity(600., 'km/s')
mass_range = 0.125
snr = 12.
Nbins = 1000.

mask = (stellar_masses > (cos_M[igal] - mass_range)) & (stellar_masses < (cos_M[igal] + mass_range)) 
mass_sample = stellar_masses[mask]
pos_sample = positions[mask]
vels_sample = vels[mask]

recession = pos_sample.in_units('kpc')*hubble
vgal_position = vels_sample + recession - vbox

#Save galaxy sample to hdf5 format
with h5py.File('./samples/cos_sample_'+str(igal)+'.h5', 'w') as hf:
	hf.create_dataset('mass', data=np.array(mass_sample))
	hf.create_dataset('position', data=np.array(pos_sample))
	hf.create_dataset('vgal_position', data=np.array(vgal_position))


for j in range(len(mass_sample)):
	spec_name = line_tri + 'cos_'+str(i)+'_gal_'+str(j)+'_'

	rolled = np.roll(range(3), -1)
	for ax in range(3): 
		"""
		v_min = vpos - vel_range; v_max = vpos + vel_range
		lambda_min = vel_to_wave(v_min, lambda_rest, c, ds.current_redshift)
		lambda_max = vel_to_wave(v_max, lambda_rest, c, ds.current_redshift)
		"""

		ray_start = pos_sample[j].copy(); ray_start[ax] = ds.domain_left_edge[ax]; ray_start[rolled[ax]] += cos_rho[igal]
		ray_end = pos_sample[j].copy(); ray_end[ax] = ds.domain_right_edge[ax]; ray_end[rolled[ax]] += cos_rho[igal]
		generate_trident_spectrum(ds, line_tri, ray_start, ray_end, spec_name+str(ax)+'_plus', lambda_rest, vgal_position[j][ax], Nbins)

		ray_start = pos_sample[j].copy(); ray_start[ax] = ds.domain_left_edge[ax]; ray_start[rolled[ax]] -= cos_rho[igal]
		ray_end = pos_sample[j].copy(); ray_end[ax] = ds.domain_right_edge[ax]; ray_end[rolled[ax]] -= cos_rho[igal]
                generate_trident_spectrum(ds, line_tri, ray_start, ray_end, spec_name+str(ax)+'_minus', lambda_rest, vgal_position[j][ax], Nbins)
