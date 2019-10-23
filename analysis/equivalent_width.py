# script to compute equivalent widths - lets keep it general

# take in list of .h5 files
# give it an ion name 
# separate equvilanet width function

# plots:
#   for each galaxy, each direction - do overlaid spectrum showing decrements of different ions
#   for each ion, 
#       equivalent width as a function of impact parameter
#       M* against SFR with coloured by EW of different ions


import h5py
import numpy as np
import sys

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)

spectra_folder = sys.argv[1]
cos_survey = sys.argv[2]
ew_file = sys.argv[3]

model = 'm50n512'
wind = 's50j7k'
snap = '151'

if cos_survey == 'cos_halos':
    pass
elif cos_survey == 'cos_dwarfs':
    pass
cos_sample_file = '/home/sapple/cgm/cos_samples/'+cos_survey+'/samples/'+model+'_'+wind+'_'+cos_survey+'_galaxy_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    gal_ids = f['gal_ids'][:]
    vgal_position = f['vgal_position'][:][:, 2]


# possible ions to choose:
ions = ['H1215', 'MgII2796', 'SiII1260', 'CIV1548', 'OVI1031', 'NeVIII770']
orients = ['x_minus', 'x_plus', 'y_minus', 'y_plus']
velocity_width = 300. #km/s
bin_size = 6.935 # km/s 
z = 0.
c = 2.98e8 # km/s

# for each galaxy:
#   find their spectra files
#   read in spectra files, extract the spectrum we want for the ion
#   get the velocity array and noise
#   get their velocity position
#   select a velocity range we want -> +/- 300km/s

for ion in ions:
    
    ew = np.zeros((len(gal_ids), 4))

    for i, gal in enumerate(gal_ids):
        
        vgal = vgal_position[i]

        for j, orient in enumerate(orients):

            # generate noise here with snr = 30.
    
            spectra_file = spectra_folder+'sample_galaxy_'+str(gal)+'_'+orient+'.h5'

            with h5py.File(spectra_file, 'r') as f:
                flux = f[ion+'_flux'][:] + f['noise'][:]
                wavelength = f['wavelength'][:]
                velocity = f['velocity'][:]

            pixel_size = wavelength[1] - wavelength[0]
            vgal_mask = (velocity < vgal + velocity_width) & (velocity > vgal - velocity_width)
            flux_use = flux[vgal_mask]
            ew[i][j] = equivalent_width(flux_use, pixel_size)

    with h5py.File(ew_file, 'a') as f:
        f.create_dataset(ion+'_ew', data=np.array(ew.flatten()))
            
