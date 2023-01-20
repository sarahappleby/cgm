# Starting from the snapfile containing only LOS particles, save out a new snapfile containing only particles
# within a temperature range (used for the collisional ionisation contribution tests).

import numpy as np
from pygadgetreader import readsnap
from save_new_dataset import make_new_dataset, prepare_out_file  


if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    verbose = 2
    minT = 6.0
    
    sample_dir = f'/disk04/sapple/data/samples/'
    snapfile = f'{sample_dir}{model}_{wind}_{snap}.hdf5'
    output_file = f'{sample_dir}{model}_{wind}_{snap}_minT_{minT}.hdf5'

    gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
    temp_mask = (gas_temp > 10**minT)

    plist = np.arange(len(gas_temp))[temp_mask]

    numpart = np.zeros(6, dtype='int')
    numpart[0] = len(plist)

    prepare_out_file(snapfile, output_file, numpart)
    make_new_dataset(snapfile, output_file, plist, verbose)
