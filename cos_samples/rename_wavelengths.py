import h5py
import numpy as np
import glob

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

model = 'm50n512'
wind = 's50noagn'
survey = 'dwarfs'

if survey == 'dwarfs':
    z = 0.
elif survey == 'halos':
    z = 0.2

c = 2.99792e5 # km/s
ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
rest_w = [1215.6701, 2796.352, 1206.500, 1548.195, 1031.927, 770.409]

spectra_folder = model +'/cos_'+survey+'/'+wind+'/spectra/'
spectra_files = sorted(glob.glob(spectra_folder+'*'))


for i, ion in enumerate(ions):

    for spectra_file in spectra_files:

        with h5py.File(spectra_file, 'a') as sf:
            if not ion + '_wavelength' in list(sf.keys()):
                
                vels = sf['velocity'][:]

                wave = vel_to_wave(vels, rest_w[i], c, z)
                sf.create_dataset(ion+'_wavelength', data=np.array(wave))
                if i == 0:
                    del sf['wavelength']

