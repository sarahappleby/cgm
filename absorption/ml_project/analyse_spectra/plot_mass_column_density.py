import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_ssfr = sf['ssfr'][:]

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    for line in lines:
        with h5py.File(f'{sample_dir}fit_column_densities_{line}.h5'):

