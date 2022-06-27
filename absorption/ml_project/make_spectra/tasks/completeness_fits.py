# Perform the fitting again for a higher SNR; save out the new fits only but not the spectra with new noise added.

import numpy as np
import pygad as pg
import sys
import os
from spectrum import Spectrum
from generate_spectra import write_spectrum

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    i = int(sys.argv[1])

    vel_range = 600. #km/s
    chisq_asym_thresh = -3. 
    snr = 100.
    sigma_noise = 1./snr
    orients = ['0_deg', '180_deg']
    pixel_size = 2.5 # km/s
    LSF = 'STIS_E140M'
    fit_contin = True

    completeness_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}_completeness/'
    spec_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    spec_file = sorted(os.listdir(spec_dir))[i]
    items = spec_file.split('_')
    orient = f'{items[4]}_{items[5]}'

    if orient in orients:
        
        spec = Spectrum(f'{spec_dir}{spec_file}')
        if os.path.isfile(f'{completeness_dir}{spec_file}'):
            sys.exit()

        line = items[3]

        spec.noise = np.random.normal(0.0, sigma_noise, len(spec.wavelengths))
        noise_vector = np.asarray([sigma_noise] * len(spec.noise))
        spec.fluxes = np.exp(-np.array(spec.taus)) + spec.noise

        if not line == 'MgII2796':
            if LSF is not None and spec.wavelengths[0] > 900:
                spec.fluxes,noise_vector = pg.analysis.absorption_spectra.apply_LSF(spec.wavelengths, spec.fluxes, noise_vector, grating=LSF)
        else:
            from astropy.convolution import convolve, Gaussian1DKernel
            fwhm = 6. / pixel_size # 6km s^-1, in pixels
            gauss_kernel = Gaussian1DKernel(stddev=fwhm / 2.355)
            spec.fluxes = convolve(spec.fluxes, gauss_kernel, boundary="wrap")
            noise_vector = convolve(noise_vector, gauss_kernel, boundary="wrap")

        if fit_contin:
            spec.continuum = pg.analysis.absorption_spectra.fit_continuum(spec.wavelengths, spec.fluxes, noise_vector, order=0, sigma_lim=1.5)
            spec.fluxes = spec.fluxes/spec.continuum

        spec.spectrum_file = f'{completeness_dir}{spec_file}'
        spec.main(vel_range=vel_range, chisq_asym_thresh=-3., write_lines=True)

    else:
        sys.exit()
