### Routine to prepare the input dataset for the absorption lines -> physical conditions ML test

import h5py
import numpy as np
import pandas as pd
import pygad as pg
import pickle
import sys

np.random.seed(1)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    line = sys.argv[4]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    chisq_lim = [3.5, 39.8, 15.8, 35.5, 6.3, 4.]
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]

    # Compute the mean cosmic mass density, for converting densities into overdensities
    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    sample_files = [f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5',
                    f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_extras.h5']

    gal_ids = []
    mass = []
    ssfr = []
    kappa_rot = []
    
    for sample_file in sample_files:
        with h5py.File(sample_file, 'r') as sf:
            gal_ids.extend(sf['gal_ids'][:])
            mass.extend(sf['mass'][:])
            ssfr.extend(sf['ssfr'][:])
            kappa_rot.extend(sf['kappa_rot'][:])
   
    gal_ids = np.array(gal_ids)
    mass = np.array(mass)
    ssfr = np.array(ssfr)
    kappa_rot = np.array(kappa_rot)

    line_files = [f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5',
                  f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}_extras.h5']

    dataset = {}
    dataset['rho'] = []
    dataset['T'] = []
    dataset['Z'] = []
    dataset['N'] = []
    dataset['b'] = []
    dataset['EW'] = []
    dataset['chisq'] = []
    dataset['dv'] = []
    dataset['gal_ids'] = []
    dataset['r_perp'] = []
   
    for line_file in line_files:

        for i in range(len(fr200)):
            with h5py.File(line_file, 'r') as hf:
                dataset['rho'].extend(hf[f'log_rho_{fr200[i]}r200'][:])
                dataset['T'].extend(hf[f'log_T_{fr200[i]}r200'][:])
                dataset['Z'].extend(hf[f'log_Z_{fr200[i]}r200'][:])
                dataset['N'].extend(hf[f'log_N_{fr200[i]}r200'][:])
                dataset['b'].extend(hf[f'b_{fr200[i]}r200'][:])
                dataset['EW'].extend(hf[f'ew_{fr200[i]}r200'][:])
                dataset['chisq'].extend(hf[f'chisq_{fr200[i]}r200'][:])
                dataset['dv'].extend(hf[f'pos_dv_{fr200[i]}r200'][:])
                dataset['gal_ids'].extend(hf[f'ids_{fr200[i]}r200'][:])
                dataset['r_perp'].extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])

    mask = (dataset['N'] > N_min[lines.index(line)]) * (dataset['chisq'] < chisq_lim[lines.index(line)]) * (dataset['b'] > 0)
    for key in dataset.keys():
        dataset[key] = dataset[key][mask]

    idx = np.array([np.where(gal_ids == l)[0][0] for l in dataset['gal_ids']]).flatten() 
    dataset['mass'] = mass[idx]
    dataset['ssfr'] = ssfr[idx]
    dataset['kappa_rot'] = kappa_rot[idx]

    dataset['delta_rho'] = dataset['rho'] - np.log10(cosmic_rho)

    # Step 2) treat the data such that unphysical/awkward values are taken care of
    dataset['EW'] = np.log10(dataset['EW'] + 1e-3)
    dataset['b'] = np.log10(dataset['b'] + 1)
    df_full = pd.DataFrame(dataset); del dataset
     
    # Step 3) Scale the data such that means are zero and variance is 1
    split = 0.8
    np.random.seed(1)
    train = np.random.rand(len(df_full)) < split
    df_full['train_mask'] = train

    print("train / test:", np.sum(train), np.sum(~train))
    df_full.to_csv(f'data/{model}_{wind}_{snap}_{line}_lines.csv')
