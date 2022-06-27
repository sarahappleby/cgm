# Deleting the fits that already exist in the spectrum files to run the fitting again.

import os
import sys
import h5py

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    spec_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    spec_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/satellites/{model}_{wind}_{snap}/log_frad_1.0/'
    spec_files = sorted(os.listdir(spec_dir))

    for spec_file in spec_files:
        print(spec_file)
        with h5py.File(f'{spec_dir}{spec_file}', 'a') as hf:
            if 'line_list' in hf.keys():
                del hf['line_list']


