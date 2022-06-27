# spectrum generation from pygad vs trident

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import gc

import pygad as pg
import trident
from yt.utilities.cosmology import Cosmology
from pygadgetreader import readsnap

from physics import wave_to_vel, vel_to_wave
from utils import read_h5_into_dict


def write_spectrum(spec_name, line, los, lambda_rest, gal_vel_pos, redshift, snr, spectrum):

    if len(los) == 2:
        los = np.append(np.array(los), -1.0)  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z. 

    with h5py.File(spec_name, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("gal_velocity_pos", data=np.array(gal_vel_pos))
        hf.create_dataset("LOS_pos", data=np.array(los))
        hf.create_dataset("redshift", data=redshift)
        hf.create_dataset("snr", data=snr)
        for k in spectrum.keys():
            hf.create_dataset(k, data=np.array(spectrum[k]))


def generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, spec_name, LSF=None, fit_contin=False, min_restr_column=5):
    if os.path.isfile(f'{spec_name}.h5'):
        return

    print('Generating pygad spectrum for ' + line)

    c = pg.physics.c.in_units_of('km/s')
    H = s.cosmology.H(s.redshift).in_units_of('(km/s)/Mpc', subs=s)
    box_width = s.boxsize.in_units_of('Mpc', subs=s)
    vbox = H * box_width
    if periodic_vel:
        v_limits = [-0.5*vbox, 1.5*vbox]
    else:
        v_limits = [0, vbox]
    total_v = np.sum(np.abs(v_limits))
    Nbins = int(np.rint(total_v / pixel_size))

    spectrum = {}
    spectrum['taus'], spectrum['col_density'], spectrum['phys_density'], spectrum['temperature'], spectrum['metallicity'], spectrum['vpec'], v_edges, spectrum['restr_column'] = \
                pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line, v_limits, Nbins=(1+int(periodic_vel))*Nbins, return_los_phys=True)
    spectrum['velocities'] = 0.5 * (v_edges[1:] + v_edges[:-1])

    if periodic_vel:
        npix_periodic = int( vbox / (max(spectrum['velocities'])-min(spectrum['velocities'])) * len(spectrum['taus']) )
        print('periodically wrapping optical depths with %d pixels'%npix_periodic)
        for i in range(0,len(spectrum['taus'])):
            if spectrum['velocities'][i] < -vbox: spectrum['taus'][i+npix_periodic] += spectrum['taus'][i]
            if spectrum['velocities'][i] > 0: spectrum['taus'][i-npix_periodic] += spectrum['taus'][i]
        # clip spectrum to original vbox
        i0 = (np.abs(spectrum['velocities'])).argmin()
        i1 = (np.abs(spectrum['velocities']-vbox)).argmin()
        spectrum['velocities'] = spectrum['velocities'][i0:i1]
        spectrum['taus'] = spectrum['taus'][i0:i1]
        spectrum['col_density'] = spectrum['col_density'][i0:i1]
        spectrum['phys_density'] = spectrum['phys_density'][i0:i1]
        spectrum['temperature'] = spectrum['temperature'][i0:i1]
        spectrum['metallicity'] = spectrum['metallicity'][i0:i1]
        spectrum['vpec'] = spectrum['vpec'][i0:i1]

    spectrum['wavelengths'] = lambda_rest * (s.redshift + 1) * (1 + spectrum['velocities'] / c)
    spectrum['metallicity'] = np.log10(np.where(spectrum['metallicity']<1.e-6,1.e-6,spectrum['metallicity']))

    sigma_noise = 1.0/snr
    spectrum['noise'] = np.random.normal(0.0, sigma_noise, len(spectrum['wavelengths']))
    noise_vector = np.asarray([sigma_noise] * len(spectrum['noise']))
    spectrum['fluxes'] = np.exp(-np.array(spectrum['taus'])) + spectrum['noise']

    if not line == 'MgII2796':
        if LSF is not None and spectrum['wavelengths'][0] > 900:
            spectrum['fluxes'],noise_vector = pg.analysis.absorption_spectra.apply_LSF(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, grating=LSF)
    else:
        from astropy.convolution import convolve, Gaussian1DKernel
        fwhm = 6. / pixel_size # 6km s^-1, in pixels
        gauss_kernel = Gaussian1DKernel(stddev=fwhm / 2.355)
        spectrum['fluxes'] = convolve(spectrum['fluxes'], gauss_kernel, boundary="wrap")
        noise_vector = convolve(noise_vector, gauss_kernel, boundary="wrap")

    if fit_contin:
        spectrum['continuum'] = pg.analysis.absorption_spectra.fit_continuum(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, order=0, sigma_lim=1.5)
        spectrum['fluxes'] = spectrum['fluxes']/spectrum['continuum']

    write_spectrum(f'{spec_name}.h5', line, los, lambda_rest, gal_vel_pos, s.redshift, snr, spectrum)

    return


def generate_trident_spectrum(ds, line_name, los, spec_name, lambda_rest, snr, Nbins, periodic_vel=True):
    if os.path.isfile(f'{spec_name}.h5'):
        return

    c = 2.997925e+05 # km/s
    print("Generating trident spectrum...",los)
    # Generate ray through box
    ray_start = [los[0], los[1], ds.domain_left_edge[2]]
    ray_end = [los[0], los[1], ds.domain_right_edge[2]]
    line_list = [line_name]
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename=f"{spec_name}_ray.h5",
                                  lines=line_list,
                                  ftype='PartType0',
                                  line_database='/disk04/sapple/trident/trident/data/line_lists/lines.txt')
    ar = ray.all_data()
    print('ray=',ray,ar)

    # Set up trident spectrum generator and generate specific spectrum using that ray
    co = Cosmology()
    print(co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc"),ds.hubble_constant)
    vbox = ds.domain_right_edge[2].in_units('kpc/h') * co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc") / ds.hubble_constant / (1.+ds.current_redshift)
    vbox_buffer = 0.5
    if periodic_vel:
        lambda_min = lambda_rest * (1 + ds.current_redshift) * (1. - (1.+vbox_buffer)*vbox.value / c)
        lambda_max = lambda_rest * (1 + ds.current_redshift) * (1. - (-vbox_buffer)*vbox.value / c)
    else:
        lambda_min = lambda_rest * (1 + ds.current_redshift) * (1. - 1.0*vbox.value / c)
        lambda_max = lambda_rest * (1 + ds.current_redshift) * (1. + 0.0*vbox.value / c)
    print('vbox=',vbox,lambda_rest,lambda_min,lambda_max)
    sg = trident.SpectrumGenerator(lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=Nbins)
    #sg = trident.SpectrumGenerator('COS-G130M')  # convolves with COS line spread fcn, gives COS resolution
    sg.make_spectrum(ray, lines=line_list)
    # Get fields and convert wavelengths to velocities. Note that the velocities are negative compared to pygad!
    wavelengths = np.array(sg.lambda_field)
    taus = np.array(sg.tau_field)
    velocities = c * ((wavelengths / lambda_rest) / (1.0 + ds.current_redshift) - 1.0)
    # periodically wrap velocities into the range [-vbox,0] -- should only be used when generating *random* LOS
    if periodic_vel:
        npix_periodic = int( vbox / ((2*vbox_buffer+1)*vbox * len(wavelengths)) )
        print('periodically wrapping optical depths with %d pixels'%npix_periodic)
        for i in range(0,len(wavelengths)):
            if velocities[i] < -vbox: taus[i+npix_periodic] += taus[i]
            if velocities[i] > 0: taus[i-npix_periodic] += taus[i]
            if velocities[i] > vbox: taus[i-npix_periodic] -= taus[i]
    # add noise
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    noise_vector = [sigma_noise] * len(noise)
    fluxes = np.array(sg.flux_field) + noise
    print('Trident generated spectrum v=',min(velocities),max(velocities),len(wavelengths),len(velocities))

    # periodically wrap velocities into the range [-vbox,0] -- should only be used when generating *random* LOS
    if periodic_vel:
        npix_periodic = int( vbox / (max(velocities)-min(velocities)) * len(wavelengths) )
        print('periodically wrapping optical depths with %d pixels'%npix_periodic)
        for i in range(0,len(wavelengths)):
            if velocities[i] < -vbox: taus[i+npix_periodic] += taus[i]
            if velocities[i] > 0: taus[i-npix_periodic] += taus[i]
    #plt.plot(velocities,np.log10(taus),'-',c='r',label='Trident')
        fluxes = np.exp(-taus) + noise

    #Save spectrum to hdf5 format
    with h5py.File('{}.h5'.format(spec_name), 'w') as hf:
        hf.create_dataset('velocities', data=np.array(velocities))
        hf.create_dataset('fluxes', data=np.array(fluxes))
        hf.create_dataset('wavelengths', data=np.array(wavelengths))
        hf.create_dataset('taus', data=np.array(taus))
        hf.create_dataset('noise', data=np.array(noise_vector))

