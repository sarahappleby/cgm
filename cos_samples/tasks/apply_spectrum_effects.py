import h5py
import numpy as np
import glob
import pygad as pg

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

model = 'm100n1024'
wind = 's50'
survey = 'halos'
snr = 12.
sigma_noise = 1.0 / snr

if survey == 'dwarfs':
    z = 0.
elif survey == 'halos':
    z = 0.2

c = 2.99792e5 # km/s
ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
rest_w = [1215.6701, 2796.352, 1206.500, 1548.195, 1031.927, 770.409]

ions = ['MgII2796']
rest_w = [2796.352]

spectra_folder = model +'/cos_'+survey+'/spectra/'
#spectra_folder = model +'/cos_'+survey+'/spectra/'
spectra_files = sorted(glob.glob(spectra_folder+'*'))

for spectra_file in spectra_files:

    for i, ion in enumerate(ions):

        with h5py.File(spectra_file, 'a') as sf:
            
            if ion+'_flux_effects' in list(sf.keys()):
                del sf[ion+'_flux_effects']
                del sf[ion+'_noise']

            waves = sf[ion+'_wavelength'][:]
            flux = sf[ion+'_flux'][:]

        noise = np.random.normal(0.0, sigma_noise, len(waves))
        noise_vector = np.asarray([sigma_noise] * len(noise))
       
        if not ion == 'MgII2796':
            f_conv, n_conv = pg.analysis.absorption_spectra.apply_LSF(waves, flux, noise_vector, grating='COS_G130M')
        else:
            f_conv = flux.copy()
        f_conv += noise
        contin = pg.analysis.absorption_spectra.fit_continuum(waves, f_conv, noise_vector, order=1, sigma_lim=1.5)
        fluxes = f_conv / contin

        with h5py.File(spectra_file, 'a') as sf:
            sf.create_dataset(ion+'_flux_effects', data=np.array(fluxes))
            sf.create_dataset(ion+'_noise', data=np.array(noise))



