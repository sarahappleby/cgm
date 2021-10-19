# Calling pygad to generate synthetic spectra.

import matplotlib.pyplot as plt
import numpy as np 
import pygad as pg
import gc
import os
import h5py


c = pg.physics.c.in_units_of('km/s')

def t_elapsed(): 
    return np.round(time.time()-TINIT,2)


def vel_to_wave(vel, lambda_rest, c, z):
    return lambda_rest * (1.0 + z) * (vel / c + 1.)


def wave_to_vel(wave, lambda_rest, c, z):
    return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)


def write_spectrum(spec_name, line, los, lambda_rest, gal_vel_pos, redshift, spectrum):

    if len(los) == 2: 
        los = np.append(np.array(los), -1.0)  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z. 

    with h5py.File(f'{spec_name}.h5', "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("gal_velocity_pos", data=np.array(gal_vel_pos))
        hf.create_dataset("LOS_pos", data=np.array(los))
        hf.create_dataset("redshift", data=redshift)
        for k in spectrum.keys():
            hf.create_dataset(k, data=np.array(spectrum[k]))


def generate_pygad_spectrum(s, los, line, lambda_rest, gal_vel_pos, periodic_vel, pixel_size, snr, spec_name, save_dir, LSF=None):
    if os.path.isfile(f'{save_dir}{spec_name}.h5'):
        with h5py.File(f'{save_dir}{spec_name}.h5', 'r') as check:
            if line + '_col_densities' in check.keys(): return
    
    print('Generating pygad spectrum for ' + line)

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
    if fit_contin:
        contin = pg.analysis.absorption_spectra.fit_continuum(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, order=0, sigma_lim=1.5)
        spectrum['fluxes'] = spectrum['fluxes']/contin
        noise_vector = noise_vector/contin

    print(f'Pygad generated spectrum v=[{velocities[0]},{velocities[-1]}] lam=[{wavelengths[0]},{wavelengths[-1]}]\ntaumax={max(taus)}\nfluxmin={min(fluxes)}\nt={t_elapsed()} s')

    write_spectrum(spec_name, line, los, lambda_rest, gal_vel_pos, s.redshift, spectrum)

    return


def extend_to_continuum(spectrum, vel_range, contin_level=1., nbuffer=10):
        
    vel_mask = (spectrum['velocities'] < spectrum['gal_velocity_pos'][()] + vel_range) & (spectrum['velocities'] > spectrum['gal_velocity_pos'][()] - vel_range)
    v_start, v_end = np.where(vel_mask)[0][0], np.where(vel_mask)[0][-1]
    
    continuum = False
    i = 0
    while not continuum:
        flux = spectrum['fluxes'][v_start - i:v_start -i +nbuffer]
        if np.abs(np.median(flux) - contin_level) / contin_level > 0.1:
            i += 1
        else:
            continuum = True

    continuum = False
    j = 0
    while not continuum:
        flux = spectrum['fluxes'][v_end + j - nbuffer: v_end +j]
        if np.abs(np.median(flux) - contin_level) / contin_level > 0.1:
            j += 1
        else:
            continuum = True

    extended_indices = np.arange(v_start - i - nbuffer, v_end+j+ nbuffer+1, 1)
    mask = np.zeros(len(vel_mask)).astype(bool)
    mask[extended_indices] = True

    return mask


def fit_spectrum(spec_name, vel_range, nbuffer=20, contin_level=1.):
   
    spectrum = {}
    with h5py.File(f'{spec_name}.h5', 'r') as f:
        for k in f.keys():
            if len(f[k].shape) == 0:
                spectrum[k] = f[k][()]
            else:
                spectrum[k] = f[k][:]
            for attr_k in f[k].attrs.keys():
                spectrum[attr_k] = f[k].attrs[attr_k]

    vel_mask = extend_to_continuum(spectrum, vel_range, contin_level)
    
    line_list = pg.analysis.fit_profiles(spectrum['ion_name'], spectrum['wavelengths'][vel_mask], spectrum['fluxes'][vel_mask], spectrum['noise'][vel_mask], 
                                         chisq_lim=2.0, max_lines=7, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
    pg.analysis.write_lines(spec_filename, line_list, istart)
    
    model_flux, N, dN, b, db, l, dl, EW = pg.analysis.plot_fit(ax, spectrum['wavelengths'][vel_mask], spectrum['fluxes'][vel_mask], spectrum['noise'][vel_mask], 
                    line_list, spectrum['ion_name'], show_plot=False)

    # throw away components with a position outside the velocity window
    # save the components
