import h5py
import numpy as np
import sys
sys.path.append('/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/')
from ignore_gals import *

from physics import vel_to_wave, equivalent_width

if __name__ == '__main__':

    cos_survey = sys.argv[1]
    model = sys.argv[2]
    wind = sys.argv[3]

    # possible ions to choose:
    ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg']
    velocity_width = 300. #km/s
    bin_size = 6. # km/s 
    c = 2.98e8 # km/s
    ngals_each = 5
    background = 'uvb_fg20'

    if cos_survey == 'dwarfs':
        snap = '151'
    elif cos_survey == 'halos':
        snap = '137'
   
    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/'
    ew_file = basic_dir+'absorption_analysis/data/cos_'+cos_survey+'_'+model + '_'+wind+'_'+snap+'_'+background+'_ew_data_lsf.h5'
    spectra_folder = basic_dir+'cos_samples/'+model+'/cos_'+cos_survey+'/'+wind+'/'+background+'/spectra/'
    cos_sample_file = basic_dir+'cos_samples/'+model+'/cos_'+cos_survey+'/samples/'+model+'_'+wind+'_cos_'+cos_survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        gal_ids = f['gal_ids'][:]
        vgal_position = f['vgal_position'][:][:, 2]

    # ignore the galaxies that dont have Simba counterparts
    ignore_simba_gals, ngals_each = get_ignore_simba_gals(model, cos_survey)

    for ion in ions:
        
        ew_l = np.zeros((len(gal_ids), len(orients)))
        ew_v = np.zeros((len(gal_ids), len(orients)))

        for i, gal in enumerate(gal_ids):
        
            if i in ignore_simba_gals:
                continue

            vgal = vgal_position[i]

            for j, orient in enumerate(orients):

                # generate noise here with snr = 30.
        
                spectra_file = spectra_folder+'sample_galaxy_'+str(gal)+'_'+orient+'.h5'

                with h5py.File(spectra_file, 'r') as f:
                    flux = f[ion+'_flux_effects'][:]
                    wavelength = f[ion+'_wavelength'][:]
                    velocity = f['velocity'][:]
               
                # the COS convolve method will go through all gratings and only convolve with the 
                # wavelength-relavent part of the spectrum

                vgal_mask = (velocity < vgal + velocity_width) & (velocity > vgal - velocity_width)
                flux_use = flux[vgal_mask]
                pixel_size = velocity[1] - velocity[0]
                ew_v[i][j] = equivalent_width(flux_use, pixel_size)
                pixel_size = wavelength[1] - wavelength[0]
                ew_l[i][j] = equivalent_width(flux_use, pixel_size)

        with h5py.File(ew_file, 'a') as f:
            f.create_dataset(ion+'_wave_ew', data=np.array(ew_l.flatten()))
            f.create_dataset(ion+'_velocity_ew', data=np.array(ew_v.flatten()))
            
