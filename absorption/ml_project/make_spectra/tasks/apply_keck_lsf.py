## Apply the Keck LSF to the MgII spectra (already generated at this point; current spectrum generation pipeline now does this)

import pygad as pg
import os
import sys
import numpy as np
import h5py
from astropy.convolution import convolve, Gaussian1DKernel

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    i = int(sys.argv[4])

    minT = 4.0

    spec_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    spec_dir = f'/disk04/sapple/data/collisional/with_uvb/{model}_{wind}_{snap}_minT_{minT}/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    print(spec_file)

    if 'MgII2796' in spec_file:

        pixel_size = 2.5 # km/s
        fwhm = 6. / pixel_size # 6km s^-1, in pixels
        snr = 30.
        gauss_kernel = Gaussian1DKernel(stddev=fwhm / 2.355)

        with h5py.File(f'{spec_dir}{spec_file}', 'r') as hf:
            taus = hf['taus'][:]
            noise = hf['noise'][:]
            waves = hf['wavelengths'][:]

        sigma_noise = 1.0/snr
        noise_vector = np.asarray([sigma_noise] * len(taus))
        fluxes = np.exp(-np.array(taus)) + noise
        
        fluxes = convolve(fluxes, gauss_kernel, boundary="wrap")
        noise_vector = convolve(noise_vector, gauss_kernel, boundary="wrap")

        continuum = pg.analysis.absorption_spectra.fit_continuum(waves, fluxes, noise_vector, order=0, sigma_lim=1.5)
        fluxes /= continuum

        with h5py.File(f'{spec_dir}{spec_file}', 'a') as hf:
            del hf['fluxes'], hf['continuum']
            hf.create_dataset('fluxes', data=np.array(fluxes))
            hf.create_dataset('continuum', data=np.array(continuum))

    else:
        sys.exit()
