# Filter out wind particles from the LOS snapshots I'd already saved.

import h5py
import caesar
import numpy as np
from pygadgetreader import readsnap

def make_new_dataset(snapfile, output_file, plist, verbose):
    ignore_fields = []
    with h5py.File(snapfile, 'r') as in_file:

        for ptype in ['PartType0']:

            pidx = int(ptype[8:]) # get particle type index

            for k in in_file[ptype]: # loop through fields

                if k in ignore_fields:
                    if verbose > 1: print(k, ' skipped...')
                    continue

                if verbose > 1: print(ptype, k)

                # load a given field (the bottleneck)
                temp_dset = in_file[ptype][k][:]
                if verbose > 1: print(temp_dset.shape)


                with h5py.File(output_file, 'a') as out_file:

                    if '%s/%s'%(ptype,k) in output_file:
                        if verbose > 1: print("dataset already exists. replacing...")
                        del output_file[ptype][k]

                    out_file[ptype][k] = temp_dset[plist]

                    temp = out_file['Header'].attrs['NumPart_ThisFile']
                    temp[pidx] = len(plist)
                    out_file['Header'].attrs['NumPart_ThisFile'] = temp

                    temp = out_file['Header'].attrs['NumPart_Total']
                    temp[pidx] = len(plist)
                    out_file['Header'].attrs['NumPart_Total'] = temp


def prepare_out_file(snapfile, output_file, numpart):
    with h5py.File(snapfile, 'r') as in_file:
        header = in_file['Header']

        with h5py.File(output_file, 'a') as out_file:
            if 'Header' not in out_file:
                out_file.copy(header, 'Header')
                out_file['Header'].attrs['NumPart_ThisFile'] = numpart
                out_file['Header'].attrs['NumPart_Total'] = numpart
            for group in ['PartType0']:
                if group not in out_file:
                    out_file.create_group(group)

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    verbose = 2

    data_dir = f'/home/rad/data/{model}/{wind}/'
    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'

    gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
    wind_particles = np.where(gas_delaytime > 0.)[0]

    output_dir = '/disk04/sapple/data/samples/'
    output_file = output_dir + model+'_'+wind+'_'+snap+'.hdf5'

    plist = np.array([])
    with h5py.File(f'{output_dir}{model}_{wind}_{snap}_particle_selection.h5', 'r') as f:
        for k in f.keys():
            if 'plist' in k:
                plist = np.append(plist, np.array(f[k][:], dtype='int'))

    plist = np.unique(np.sort(plist)).astype('int')

    plist_wind_ids = np.in1d(plist, wind_particles).nonzero()[0]
    plist = np.delete(plist, plist_wind_ids)

    numpart = np.zeros(6, dtype='int')
    numpart[0] = len(plist)

    prepare_out_file(snapfile, output_file, numpart)

    make_new_dataset(snapfile, output_file, plist, verbose)
