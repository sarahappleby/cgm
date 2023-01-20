### Routine to collate the overall bulk spectra properties from Appleby+22 sample, including galaxy properties

import numpy as np
import h5py
import pandas as pd
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    norients = 8
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    data_dir = '/disk04/sapple/data/normal/results/'
    sample_dir = f'/disk04/sapple/data/samples/'

    dataset = {}

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        ngals = len(sf['gal_ids'][:])
        dataset['gal_id'] = np.repeat(sf['gal_ids'][:], norients*nbins_fr200)
        dataset['mass'] = np.repeat(sf['mass'][:], norients*nbins_fr200)
        dataset['ssfr'] = np.repeat(sf['ssfr'][:], norients*nbins_fr200)
        dataset['kappa_rot'] = np.repeat(sf['kappa_rot'][:], norients*nbins_fr200)

    dataset['fr200'] = np.tile(np.repeat(fr200, norients), ngals)

    for i in range(len(lines)):
        
        dataset[f'EW_{lines_short[i]}'] = np.zeros(ngals*norients*nbins_fr200)
        dataset[f'los_rho_{lines_short[i]}'] = np.zeros(ngals*norients*nbins_fr200)
        dataset[f'los_T_{lines_short[i]}'] = np.zeros(ngals*norients*nbins_fr200)
        dataset[f'los_Z_{lines_short[i]}'] = np.zeros(ngals*norients*nbins_fr200)

        sum_ew_file = f'{data_dir}{model}_{wind}_{snap}_ew_{lines[i]}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)
        los_rhoTZ_file = f'{data_dir}{model}_{wind}_{snap}_Nweighted_rhoTZ_{lines[i]}.h5'
        los_rhoTZ_dict = read_h5_into_dict(los_rhoTZ_file)

        for j in range(ngals):
            i_start = j*norients*nbins_fr200

            for k in range(nbins_fr200):

                dataset[f'EW_{lines_short[i]}'][i_start:i_start+norients] = sum_ew_dict[f'ew_wave_{fr200[k]}r200'][j]
                dataset[f'los_rho_{lines_short[i]}'][i_start:i_start+norients] = los_rhoTZ_dict[f'los_rho_{fr200[k]}r200'][j]
                dataset[f'los_T_{lines_short[i]}'][i_start:i_start+norients] = los_rhoTZ_dict[f'los_T_{fr200[k]}r200'][j]]
                dataset[f'los_Z_{lines_short[i]}'][i_start:i_start+norients] = los_rhoTZ_dict[f'los_Z_{fr200[k]}r200'][j]

                i_start += norients

    input_df = pd.DataFrame(dataframe)
