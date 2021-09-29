import pygad as pg
import numpy as np
import h5py 
import caesar
import matplotlib.pyplot as plt
import yt

def vel_to_wave(vel, lambda_rest, c, z):
    return lambda_rest * (1.0 + z) * (vel / c + 1.)

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z - 9.


if __name__ == '__main__':

    model = 'm25n256'
    snap = 151
    wind = 's50'
    
    line = 'H1215' # see pygad/pygad/analysis/absorption_spectra.py for a list of lines in pygad to choose from
    lambda_rest = float(pg.analysis.absorption_spectra.lines[line]['l'].split()[0])
    rho = 0. # kpc/h, impact parameter away from the galaxy to take the line of sight
    pixel_size = 2.5 # spectrum pixel width in km/s
    snr = 12. # signal to noise ratio
    periodic_vel = True # do periodic wrapping on the velocities
    vel_range = 600 # km/s, cut out portion of spectrum near galaxy
    ngal = 1 # make a spectrum for the ngal-th star forming galaxy in the caesar file


    save_dir = f'/home/sarah/cgm/test/' # change this
    data_dir = f'/home/sarah/data/'
    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'
    caesarfile = f'{data_dir}{model}_{snap}.hdf5'

    s = pg.Snapshot(snapfile) # load in snapshot with pygad
    sim = caesar.load(caesarfile) # load in caesar file

    hubble = s.cosmology.H().in_units_of('km/s/kpc')
    vbox = s.boxsize.in_units_of('kpc') * hubble / (1.+s.redshift) # get right edge of the box in velocity space
    v_limits = [-1.*vel_range, float(vbox)+vel_range] # go from -600 km/s to vbox + 600 km/s
    total_v_range = np.sum(np.abs(v_limits))
    Nbins = int(np.rint(total_v_range / pixel_size))

    # find the galaxy ids of star forming galaxies - this will give us more gas around galaxies to play with for our spectra :) 
    quench = quench_thresh(s.redshift) 
    gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
    gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
    gal_ssfr = gal_sfr / gal_sm
    gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
    sf_ids = np.arange(len(sim.galaxies))[gal_ssfr > quench]
    gal_id = sf_ids[ngal]
    gal = sim.galaxies[gal_id]
    spec_name = f'galaxy_{gal_id}_{line}'

    gal_pos = np.array(gal.pos.in_units('kpc/h')) * (1+s.redshift)
    gal_vel = np.array(gal.vel.in_units('km/s'))
    gal_recession = np.array(gal.pos.in_units('kpc/h'))*float(hubble)
    gal_vpos = (gal_vel + gal_recession)[2]
    gal_r200 = float(gal.halo.virial_quantities['r200c'].in_units('kpc/h') / (1.+s.redshift))

    los = gal_pos[:2].copy(); los[0] += rho # choose a line of sight for the galaxy with [xgal + rho, ygal, 0] -> [xgal + rho, ygal, boxwidth]

    # generate the spectrum in terms of optical depths (tau)
    # get other useful line of sight quantities: column densities, physical densities, mass-weighted temperatures, metal fractions, line of sight velocities
    # (all binned along the line of sight in velocity pixels), plus the edges of the velocity bins and the column densities of all particles that contributed
    # to the spectrum
    taus, col_densities, dens, temps, metal_frac, los_vel, v_edges, restr_column = \
                pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line, v_limits, Nbins=Nbins, return_los_phys=True)

    # compute the fluxes and wavelengths of the spectrum
    fluxes = np.exp(-1.*taus)
    velocities = 0.5 * (v_edges[1:] + v_edges[:-1])
    wavelengths = vel_to_wave(velocities, lambda_rest, np.array(pg.physics.cosmology.c.in_units_of('km/s')), s.redshift)
    
    # make a noise vector to add to our fluxes
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    noise_vector = np.asarray([sigma_noise] * len(noise))

    ### here we normally periodically wrap the velocities, but I can't remember Romeel's new method for this :/

    # add in some observational effects to the fluxes: the line spread function (LSF), random noise and a continuum fit
    if not line == 'MgII2796':
        f_conv, n_conv = pg.analysis.absorption_spectra.apply_LSF(wavelengths, fluxes, noise_vector, grating='COS_G130M')
    else:
        f_conv = fluxes.copy()
    f_conv_noise = f_conv + noise
    contin = pg.analysis.absorption_spectra.fit_continuum(wavelengths, f_conv_noise, noise_vector, order=1, sigma_lim=1.5)
    f_conv_noise_contin = f_conv_noise / contin

    with h5py.File(f'{save_dir}{spec_name}.h5', 'a') as hf:
        hf.create_dataset(line+'_flux_orig', data=np.array(fluxes))
        hf.create_dataset(line+'_flux_effects', data=np.array(f_conv_noise_contin))
        hf.create_dataset(line+'_noise', data=np.array(noise))
        hf.create_dataset(line+'_tau', data=np.array(taus))
        hf.create_dataset(line+'_temp', data=np.array(temps))
        hf.create_dataset(line+'_col_dens', data=np.array(col_densities))
        hf.create_dataset(line+'_dens', data=np.array(dens))
        hf.create_dataset(line+'_metal_frac', data=np.array(metal_frac)) 
        hf.create_dataset(line+'_los_vel', data=np.array(los_vel))
        hf.create_dataset(line+'_wavelength', data=np.array(wavelengths))
        if 'velocity' not in hf.keys():
            hf.create_dataset('velocity', data=np.array(velocities))

    # make a mask for the portion of the spectrum around the galaxy
    vel_mask = (velocities > gal_vpos - vel_range) & (velocities < gal_vpos + vel_range) 

    # plot the clean spectrum (no observational effects) 
    fig, ax = plt.subplots(2, 1, figsize=(15, 6))
    ax[0].plot(velocities, taus)
    ax[0].set_ylabel(r'$\tau$')
    ax[0].axvline(gal_vpos, c='m')
    ax[1].plot(velocities, fluxes)
    ax[1].set_ylabel('Flux')
    ax[1].set_xlabel('V (km/s)')
    ax[1].set_ylim(0, 1)
    ax[1].axvline(gal_vpos, c='m')
    plt.savefig(f'{save_dir}{spec_name}.png')
    plt.clf()
