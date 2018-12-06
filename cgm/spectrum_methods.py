import pygad as pg
import numpy as np

from yt.utilities.cosmology import Cosmology


def generate_trident_spectrum(ds, line_name, ray_start, ray_end, spec_name, lambda_rest, Nbins):
    print("Generating trident spectrum...")
    # Generate ray through box
    line_list = [line_name]
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=line_list,
                                  ftype='PartType0')
    ar = ray.all_data()
    print 'ray=',ray,ar

    # Set up trident spectrum generator and generate specific spectrum using that ray
    co = Cosmology()
    print co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc"),ds.hubble_constant
    vbox = ds.domain_right_edge[2].in_units('kpc') * co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc") / ds.hubble_constant / (1.+ds.current_redshift)
    c = yt.units.c.in_units('km/s')
    lambda_min = lambda_rest * (1 + ds.current_redshift)
    lambda_max = lambda_rest * (1 + ds.current_redshift) * (1. + vbox.value / c)
    print 'vbox=',vbox,lambda_rest,lambda_min,lambda_max
    sg = trident.SpectrumGenerator(lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=Nbins)
    #sg = trident.SpectrumGenerator('COS-G130M')  # convolves with COS line spread fcn, gives COS resolution
    sg.make_spectrum(ray, lines=line_list)

    # Get fields and convert wavelengths to velocities. Note that the velocities are negative compared to pygad!
    wavelengths = np.array(sg.lambda_field)
    taus = np.array(sg.tau_field)
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    noise_vector = [sigma_noise] * len(noise)
    fluxes = np.array(sg.flux_field)
    velocities = c * ((wavelengths / lambda_rest) / (1.0 + ds.current_redshift) - 1.0)
    print 'Trident generated spectrum v=',min(velocities),max(velocities),len(wavelengths),len(velocities)

    # plot spectrum
    plt.plot(velocities,fluxes,':',c='g',label='Trident')
    #plt.plot(velocities,fluxes_orig,':',c='b',label='Trident')
    plt.plot(velocities,noise_vector,'--',c='c')
    plt.xlim(-vbox,vbox)

    #Save spectrum to hdf5 format
    with h5py.File('{}.h5'.format(spec_name), 'w') as hf:
        hf.create_dataset('velocity', data=np.array(velocities))
        hf.create_dataset('flux', data=np.array(fluxes))
        hf.create_dataset('wavelength', data=np.array(wavelengths))
        hf.create_dataset('tau', data=np.array(taus))
        hf.create_dataset('noise', data=np.array(noise_vector))
    print('Trident spectrum done [t=%g s]'%(t_elapsed()))


def get_vels(v_edges):
	"""
	Find velocities at bin centers of spectrum
	Args:
		v_edges: velocities at bin edges
	Returns:
		vels: velocities at bin centers
	"""
	vels = 0.5 * (v_edges[1:] + v_edges[:-1])
	return vels

def get_wavelengths(vels, lambda_rest, z, c):
	"""
	Find wavelengths corresponding to velocity bins at given redshift
	Args:
		vels: velocities (km/s)
		lambda_rest: rest wavelength of line
	Returns:
		wavelengths: 
	"""
	wavelengths = lambda_rest* (z+1)* (1 + vels / c)
	return wavelengths

def get_flux(taus, sigma_noise, noise=True):
	"""
	Find fluxes from optical depths, with noise sampled from Gaussian distribution
	"""
	flux = np.exp(-1.*taus)
	if noise:
		flux += np.random.normal(0.0, sigma_noise, len(taus))
	return flux

