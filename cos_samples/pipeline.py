import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import h5py

import yt
import trident
import caesar
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg

import pyigm
from pyigm.cgm import cos_halos as pch

import gc
import time

def t_elapsed(): return np.round(time.time()-TINIT,2)

def vel_to_wave(vel, lambda_rest, c, z):
	return lambda_rest * (1.0 + z) * (vel / c + 1.)

def wave_to_vel(wave, lambda_rest, c, z):
	return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)

def generate_trident_spectrum(ds, line_list, ray_start, ray_end, spec_name, lambda_rest, vpos):
    
    print("Generating trident spectrum...")
    # Generate ray through box

    TINIT = time.time()
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=line_list,
                                  ftype='PartType0')    
    print('Trident spectrum done [t=%g s]'%(np.round(time.time()-TINIT,2)))    

    #sg = trident.SpectrumGenerator(lambda_min=lambda_min.value, lambda_max=lambda_max, n_lambda=Nbins)
    sg = trident.SpectrumGenerator('COS-G130M')  # convolves with COS line spread fcn, gives COS resolution
    
    with h5py.File('./spectra/{}.h5'.format(spec_name), 'w') as hf:
	for i in range(len(line_list)):

            line = line_list[i]
            l_rest = lambda_rest[i]
            print 'Saving data for line ' + line
	    name = line.replace(' ', '_')
 
	    sg.make_spectrum(ray, lines=[line])
	    
            # Get fields and convert wavelengths to velocities. Note that the velocities are negative compared to pygad!
            taus = np.array(sg.tau_field)
            fluxes = np.array(sg.flux_field)
            wavelengths = np.array(sg.lambda_field)
            velocities = wave_to_vel(wavelengths, l_rest, c, ds.current_redshift)

            plt.plot(velocities, fluxes)
            plt.axvline(vpos, linewidth=1, c='k')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('Flux')
            plt.savefig('./plots/'+spec_name+'_'+name+'.png')
            plt.clf()
            
            #Save spectrum to hdf5 format
            hf.create_dataset(name+'_flux', data=np.array(fluxes))
            hf.create_dataset(name+'_tau', data=np.array(taus))
       
        sigma_noise = 1.0/snr
        noise = np.random.normal(0.0, sigma_noise, len(wavelengths))

        #Save spectrum to hdf5 format
        hf.create_dataset('velocity', data=np.array(velocities))
        hf.create_dataset('wavelength', data=np.array(wavelengths))
        hf.create_dataset('noise', data=np.array(noise))
  
    del ray; gc.collect()
    return

def generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, vpos, c, spec_name, save_dir):
	if os.path.isfile(save_dir+'/spectra/{}.h5'.format(spec_name)):
      		check = h5py.File(save_dir+'/spectra/{}.h5'.format(spec_name), 'r')
		if line + '_col_densities' in check.keys(): return
		check.close()
	
	if periodic_vel: v_limits = [-600., vbox.value+600.]
	else: v_limits = [0., vbox]

	print 'Generating pygad spectrum for ' + line 
	taus, col_densities, temps, v_edges, restr_column = pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line, v_limits, Nbins=Nbins)
        fluxes = np.exp(-1.*taus)
	velocities = 0.5 * (v_edges[1:] + v_edges[:-1])
	wavelengths = lambda_rest * (s.redshift + 1) * (1 + velocities / c)
	sigma_noise = 1.0/snr
	noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
		
	if periodic_vel:
		npix_periodic = int( vbox / (max(velocities)-min(velocities)) * len(wavelengths) )
		print 'periodically wrapping optical depths with %d pixels'%npix_periodic
		for i in range(0,len(wavelengths)):
			if velocities[i] < 0.: taus[i+npix_periodic] += taus[i]
			if velocities[i] > vbox: taus[i-npix_periodic] += taus[i]

	plt.plot(velocities, fluxes)
        plt.axvline(vpos, linewidth=1, c='k')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Flux')
        plt.savefig(save_dir+'/plots/'+spec_name+'_'+line+'.png')
        plt.clf()

	with h5py.File(save_dir+'/spectra/{}.h5'.format(spec_name), 'a') as hf:
	    hf.create_dataset(line+'_flux', data=np.array(fluxes))
            hf.create_dataset(line+'_tau_periodic', data=np.array(taus))
	    hf.create_dataset(line+'_temp', data=np.array(temps))
	    hf.create_dataset(line+'_col_densities', data=np.array(col_densities))
	hf.close()

	print 'Completed for ' + line
	
	with h5py.File(save_dir+'/spectra/{}.h5'.format(spec_name), 'r') as hf:
		check = hf.keys()
	if 'wavelength' not in check:
    		with h5py.File(save_dir+'/spectra/{}.h5'.format(spec_name), 'a') as hf:
    			hf.create_dataset('velocity', data=np.array(velocities))
        		hf.create_dataset('wavelength', data=np.array(wavelengths))
 	       		hf.create_dataset('noise', data=np.array(noise))
    		hf.close()
	return

model = sys.argv[1]
snap = sys.argv[2]
wind = sys.argv[3]
cos_id = int(sys.argv[4])
line = sys.argv[5]
lambda_rest = float(filter(str.isdigit, line))
snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
save_dir = '/home/sapple/cgm/cos_samples/pygad/periodic/kpch/'

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

ds = yt.load(snapfile)
s = pg.Snap(snapfile)

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)
c = pg.physics.c.in_units_of('km/s')
snr = 12.
periodic_vel = True

sigma_vel = 6. # km/s
Nbins = int(np.rint(vbox / sigma_vel))

if not os.path.isfile(save_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5'):
	sim = caesar.load(infile)
	gals = sim.central_galaxies
	stellar_masses = YTArray([gals[i].masses['stellar'].in_units('Msun') for i in range(len(gals))], 'Msun')
	sfr = np.array([gals[i].sfr.in_units('Msun/yr') for i in range(len(gals))])
	ssfr = sfr / stellar_masses
	positions = YTArray([gals[i].pos.in_units('kpc/h') for i in range(len(gals))], 'kpc/h')
	vels = YTArray([gals[i].vel.in_units('km/s') for i in range(len(gals))], 'km/s')
	stellar_masses = np.log10(stellar_masses)
	recession = positions.in_units('kpc')*hubble
	vgal_position = vels + recession

	print 'Loaded caesar galaxy data from model ' + model + ' snapshot ' + snap

	# example used in Ford et al 2016
	# COS-halos galaxy with M=10**10.2 Msun and b = 18.26 kpc = 26.85 kpc/h
	#cos_mass = 10.2
	#cos_b = 26.85

	mass_range = 0.125

	# find the galaxies in the mass range of each COS Halos galaxy
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

	choose = np.sort(np.random.choice(range(len(indices)), 5, replace=False))
	print 'Chosen galaxies ' + str(indices[choose])
	gal_ids = indices[choose]
	
	mass_sample = stellar_masses[indices[choose]]
	ssfr_sample = ssfr[indices[choose]]

	pos_sample = positions[indices[choose]]
	vels_sample = vels[indices[choose]]
	vgal_position_sample = vgal_position[indices[choose]]

	# save some memory:
	del gals, stellar_masses, ssfr, mass_mask, ssfr_mask

	with h5py.File(save_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5', 'w') as hf:
		hf.create_dataset('mask', data=np.array(mask))
		hf.create_dataset('gal_ids', data=np.array(indices[choose]))
        	hf.create_dataset('mass', data=np.array(mass_sample))
		hf.create_dataset('ssfr', data=np.array(ssfr_sample))
       		hf.create_dataset('position', data=np.array(pos_sample))
       		hf.create_dataset('vgal_position', data=np.array(vgal_position_sample))
	hf.close()

else:
	cos_sample = h5py.File(save_dir+'/samples/cos_galaxy_'+str(cos_id)+'_sample_data.h5', 'r')
	gal_ids = cos_sample['gal_ids'].value
	pos_sample = cos_sample['position'].value
	vgal_position_sample = cos_sample['vgal_position'].value
	cos_sample.close()	
	
for i in range(len(gal_ids)):
	print 'Generating spectra for sample galaxy ' + str(gal_ids[i])
	gal_name = 'cos_galaxy_'+str(cos_id)+'_sample_galaxy_' + str(gal_ids[i]) + '_'

	# for pygad:

	# pygad assumes same units as s['pos'], so need to convert from kpc/h to ckpc/h_0	
	print 'Converting galaxy positions from kpc/h to ckpc/h_0 for pygad'

	spec_name = gal_name + 'x_plus'
	los = pos_sample[i][:2].copy(); los[0] += cos_rho[cos_id].value
	print 'In kpc/h: ' + str(los)
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')	
	#print 'In ckpc/h_0: ' + str(los)
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, vgal_position_sample[i], c, spec_name, save_dir)
	
	spec_name = gal_name + 'x_minus'
	los = pos_sample[i][:2].copy(); los[0] -= cos_rho[cos_id].value
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, vgal_position_sample[i], c, spec_name, save_dir)

	spec_name = gal_name + 'y_plus'
	los = pos_sample[i][:2].copy(); los[1] += cos_rho[cos_id].value
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, vgal_position_sample[i], c, spec_name, save_dir)
	
	spec_name = gal_name + 'y_minus'	
	los = pos_sample[i][:2].copy(); los[1] -= cos_rho[cos_id].value
	#los /= ((1. + s.redshift)*co.hubble_parameter(s.redshift).in_units('km/s/kpc'))
	#los *= co.hubble_parameter(0.0).in_units('km/s/kpc')
	generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, vgal_position_sample[i], c, spec_name, save_dir)
