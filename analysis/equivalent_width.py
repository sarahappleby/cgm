# script to compute equivalent widths - lets keep it general

# take in list of .h5 files
# give it an ion name 
# separate equivalent width function

# plots:
#   for each galaxy, each direction - do overlaid spectrum showing decrements of different ions
#   for each ion, 
#       equivalent width as a function of impact parameter
#       M* against SFR with coloured by EW of different ions


import h5py
import numpy as np
import sys
import yt

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)

if __name__ == '__main__':

    cos_survey = sys.argv[1]

    model = 'm100n1024'
    wind = 's50'

    # possible ions to choose:
    ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    orients = ['x_minus', 'x_plus', 'y_minus', 'y_plus']
    velocity_width = 300. #km/s
    bin_size = 6. # km/s 
    c = 2.98e8 # km/s

    if cos_survey == 'dwarfs':
        snap = '151'
    elif cos_survey == 'halos':
        snap = '137'
    
    """
    # get size of bins 
    snapfile = '/home/rad/data/'+model+'/'+wind+'/snap_'+model+'_'+snap+'.hdf5'
    ds = yt.load(snapfile)
    co = yt.utilities.cosmology.Cosmology()
    hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
    vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)
    pixel_size = 6. # km/s
    Nbins = int(np.rint(vbox / pixel_size))
    pixel_size = vbox / Nbins # very close to 6 km/s
    """

    ew_file = '/home/sapple/cgm/analysis/data/cos_'+cos_survey+'_'+model + '_'+snap+'_ew_data.h5'
    spectra_folder = '/home/sapple/cgm/cos_samples/cos_'+cos_survey+'/spectra/'
    cos_sample_file = '/home/sapple/cgm/cos_samples/cos_'+cos_survey+'/samples/'+model+'_'+wind+'_cos_'+cos_survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        gal_ids = f['gal_ids'][:]
        vgal_position = f['vgal_position'][:][:, 2]


    # for each galaxy:
    #   find their spectra files
    #   read in spectra files, extract the spectrum we want for the ion
    #   get the velocity array and noise
    #   get their velocity position
    #   select a velocity range we want -> +/- 300km/s

    for ion in ions:
        
        ew_l = np.zeros((len(gal_ids), 4))
        ew_v = np.zeros((len(gal_ids), 4))

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
                pixel_size = velocity[1] - velocity[0]
                ew_v[i][j] = equivalent_width(flux_use, pixel_size)
                pixel_size = wavelength[1] - wavelength[0]
                ew_l[i][j] = equivalent_width(flux_use, pixel_size)

        with h5py.File(ew_file, 'a') as f:
            f.create_dataset(ion+'_wave_ew', data=np.array(ew_l.flatten()))
            f.create_dataset(ion+'_velocity_ew', data=np.array(ew_v.flatten()))
            
