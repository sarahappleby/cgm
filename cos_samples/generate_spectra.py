import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np 
import pygad as pg
import gc
import os
import h5py

def t_elapsed(): return np.round(time.time()-TINIT,2)

def vel_to_wave(vel, lambda_rest, c, z):
	return lambda_rest * (1.0 + z) * (vel / c + 1.)

def wave_to_vel(wave, lambda_rest, c, z):
	return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)

def generate_trident_spectrum(ds, line_list, ray_start, ray_end, spec_name, lambda_rest, vpos):
    import trident
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

def generate_pygad_spectrum(s, los, line, lambda_rest, vbox, periodic_vel, Nbins, snr, vpos, c, spec_name, save_dir):
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
