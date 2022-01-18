# Calling pygad to generate synthetic spectra.

import matplotlib.pyplot as plt
import numpy as np 
import pygad as pg
import gc
import os
import h5py
from pygadgetreader import readsnap


def t_elapsed(): 
    return np.round(time.time()-TINIT,2)


def vel_to_wave(vel, lambda_rest, c, z):
    return lambda_rest * (1.0 + z) * (vel / c + 1.)


def wave_to_vel(wave, lambda_rest, c, z):
    return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)


def write_spectrum(spec_name, line, los, lambda_rest, gal_vel_pos, redshift, spectrum):

    if len(los) == 2: 
        los = np.append(np.array(los), -1.0)  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z. 

    with h5py.File(spec_name, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("gal_velocity_pos", data=np.array(gal_vel_pos))
        hf.create_dataset("LOS_pos", data=np.array(los))
        hf.create_dataset("redshift", data=redshift)
        for k in spectrum.keys():
            hf.create_dataset(k, data=np.array(spectrum[k]))


def write_line_list(spectrum, spec_file):
    
    with h5py.File(spec_file, 'a') as hf:
        if 'lines' in hf.keys():
            del hf['lines']
        lines = hf.create_group("lines")
        lines.create_dataset("fit_region", data=np.array(spectrum['lines']['region']))
        lines.create_dataset("fit_logN", data=np.array(spectrum['lines']['N']))
        lines.create_dataset("fit_dlogN", data=np.array(spectrum['lines']['dN']))
        lines.create_dataset("fit_b", data=np.array(spectrum['lines']['b']))
        lines.create_dataset("fit_db", data=np.array(spectrum['lines']['db']))
        lines.create_dataset("fit_l", data=np.array(spectrum['lines']['l']))
        lines.create_dataset("fit_dl", data=np.array(spectrum['lines']['dl']))
        lines.create_dataset("fit_EW", data=np.array(spectrum['lines']['EW']))
        lines.create_dataset("fit_Chisq", data=np.array(spectrum['lines']['Chisq']))


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
    if fit_contin:
        spectrum['continuum'] = pg.analysis.absorption_spectra.fit_continuum(spectrum['wavelengths'], spectrum['fluxes'], noise_vector, order=0, sigma_lim=1.5)
        spectrum['fluxes'] = spectrum['fluxes']/spectrum['continuum']
        noise_vector = noise_vector/spectrum['continuum']

    spectrum['n90_restr_column'] = check_restr_column(s.filename, los, spectrum['restr_column'])
    if spectrum['n90_restr_column'] < min_restr_column:
        print(f'WARNING: fewer than {min_restr_column} gas elements contribute 90% of the total column density.')
    del spectrum['restr_column']

    write_spectrum(f'{spec_name}.h5', line, los, lambda_rest, gal_vel_pos, s.redshift, spectrum)

    return


def get_los_particles(los, gas_pos, hsml):

    x_dist = np.abs(los[0] - gas_pos[:, 0])
    y_dist = np.abs(los[1] - gas_pos[:, 1])
    hyp_sq = x_dist**2 + y_dist**2
    dist_mask = hyp_sq < hsml**2
    partids_los = np.arange(len(hsml))[dist_mask]
    return partids_los


def check_restr_column(snapfile, los, restr_column, f_total_column=0.9):
    
    hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving
    los_particles = get_los_particles(los, gas_pos, hsml)

    los_restr_column = restr_column[los_particles]
    total_column = np.sum(los_restr_column)
    restr_cumsum = np.cumsum(np.sort(los_restr_column)[::-1])
    try:
        i = next(i for i, x in enumerate(restr_cumsum) if x >= f_total_column * total_column)
        return i
    except:
        return -9999


def extend_to_continuum(spectrum, vel_range, contin_level=1., nbuffer=20):
        
    vel_mask = (spectrum['velocities'] < spectrum['gal_velocity_pos'][()] + vel_range) & (spectrum['velocities'] > spectrum['gal_velocity_pos'][()] - vel_range)
    v_start, v_end = np.where(vel_mask)[0][0], np.where(vel_mask)[0][-1]
    
    continuum = False
    i = 0
    while not continuum:
        flux = spectrum['fluxes'][v_start - i:v_start -i +nbuffer]
        if np.abs(np.median(flux) - contin_level) / contin_level > 0.05:
            i += 1
        else:
            continuum = True

    continuum = False
    j = 0
    while not continuum:
        flux = spectrum['fluxes'][v_end + j - nbuffer: v_end +j]
        if np.abs(np.median(flux) - contin_level) / contin_level > 0.05:
            j += 1
        else:
            continuum = True

    extended_indices = np.arange(v_start - i - nbuffer, v_end+j+ nbuffer+1, 1)
    extended_indices = np.delete(extended_indices, np.argwhere(extended_indices > len(vel_mask) -1))
    mask = np.zeros(len(vel_mask)).astype(bool)
    mask[extended_indices] = True

    return mask


def read_h5py_into_dict(h5file):
    data = {}
    with h5py.File(h5file, 'r') as f:
        for k in f.keys():

            if type(f[k]) == h5py._hl.group.Group:
                data[k] = {}

                for gk in f[k].keys():

                    if len(f[k][gk].shape) == 0:
                        data[k][gk] = f[k][gk][()]
                    else:
                        data[k][gk] = f[k][gk][:]

            else:      
                if len(f[k].shape) == 0:
                    data[k] = f[k][()]
                else:
                    data[k] = f[k][:]
            
                for attr_k in f[k].attrs.keys():
                    data[attr_k] = f[k].attrs[attr_k]
    
    return data


def fit_spectrum(spec_file, vel_range=600., nbuffer=20, z=None):
   
    spectrum = read_h5py_into_dict(spec_file)
    if 'lines' in spectrum.keys():
        return

    contin_level = spectrum['continuum'][0]
    vel_mask = extend_to_continuum(spectrum, vel_range, contin_level)

    spectrum['lines'] = pg.analysis.fit_profiles(spectrum['ion_name'], spectrum['wavelengths'][vel_mask], spectrum['fluxes'][vel_mask], spectrum['noise'][vel_mask], 
                                         chisq_lim=2.5, max_lines=10, logN_bounds=[12,17], b_bounds=[3,100], mode='Voigt')
    
    line_velocities = wave_to_vel(spectrum['lines']['l'], spectrum['lambda_rest'], pg.physics.c.in_units_of('km/s'), z)

    outwith_vel_mask = ~((line_velocities > spectrum['gal_velocity_pos'] - vel_range) & (line_velocities < spectrum['gal_velocity_pos'] + vel_range))

    for k in spectrum['lines'].keys():
        spectrum['lines'][k] = np.delete(spectrum['lines'][k], outwith_vel_mask)

    write_line_list(spectrum, spec_file)

def plot_fit(spec_file):

    spectrum = read_h5py_into_dict(spec_file) 

    contin_level = spectrum['continuum'][0]
    vel_mask = extend_to_continuum(spectrum, vel_range, contin_level)

    spectrum['flux_model'] = get_flux_model(spectrum['wavelengths'], spectrum['lines'], spectrum['ion_name'])

    regions, indices =  pg.analysis.find_regions(spectrum['wavelengths'][vel_mask], spectrum['fluxes'][vel_mask], spectrum['noise'][vel_mask])

    line_velocities = wave_to_vel(regions, spectrum['lambda_rest'], spectrum['redshift'])

    plt.plot(spectrum['velocities'][vel_mask], spectrum['fluxes'][vel_mask], label='data')
    plt.plot(spectrum['velocities'][vel_mask], spectrum['fluxes_model'][vel_mask], label='model')
    for r in line_velocities:
         plt.axvline(r[0], c='g', ls='--', lw=0.75)
         plt.axvline(r[1], c='g', ls='--', lw=0.75)
    plt.legend()
    plt.savefig(spec_file.replace('.h5', '.png'))
    plt.clf()
