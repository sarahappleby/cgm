import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *
from spectrum import Spectrum

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    fr200 = sys.argv[4]
    line = sys.argv[5]

    vel_range = 600. #km/s
    orients = ['0_deg', '45_deg', '90_deg', '135_deg', '180_deg', '225_deg', '270_deg', '315_deg'] 

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_Nregions_{line}.h5'

    s = pg.Snapshot(f'{sample_dir}{model}_{wind}_{snap}.hdf5')
    redshift = s.redshift

    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'

    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    all_ewspec = []
    all_ewfit = []
    all_chisq = []

    for i in range(len(gal_ids)):
        for o, orient in enumerate(orients):

            spec_name = f'sample_galaxy_{gal_ids[i]}_{line}_{orient}_{fr200}r200'
            spec = Spectrum(f'{spectra_dir}{spec_name}.h5') 
            
            spec.prepare_spectrum(vel_range=vel_range) 
            regions_l, regions_i = pg.analysis.find_regions(spec.waves_fit, spec.fluxes_fit, spec.noise_fit, min_region_width=2, extend=True)
            regions_v = np.array(wave_to_vel(regions_l, spec.lambda_rest, redshift))

            pixel_size = spec.wavelengths[1] - spec.wavelengths[0]

            if hasattr(spec, 'line_list'):
                
                for j in range(len(regions_v)):
                
                    spec_mask = (spec.velocities > regions_v[j][0]) & (spec.velocities < regions_v[j][1])
                    all_ewspec.append(np.log10(equivalent_width(spec.fluxes[spec_mask], pixel_size)))
                
                    fit_mask = spec.line_list['region'] == j
                    all_ewfit.append(np.log10(np.nansum(spec.line_list['EW'][fit_mask])))

                    all_chisq.append(spec.line_list['Chisq'][fit_mask][0])

    with h5py.File(results_file, 'a') as hf:
        if not f'ewspec_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ewspec_{fr200}r200', data=np.array(all_ewspec))
        if not f'ewfit_{fr200}r200' in hf.keys():
            hf.create_dataset(f'ewfit_{fr200}r200', data=np.array(all_ewfit))
