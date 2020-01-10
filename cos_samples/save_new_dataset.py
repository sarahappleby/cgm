# Script for creating subsets of particles from gizmo sims
# Based on scripts by Sydney Lower and Chris Lovell
# https://gist.github.com/christopherlovell/5a504c2c9d26efb6e073324d80c755a6 

import h5py
import caesar
import numpy as np
import json

model = 'm100n1024'
wind = 's50'
survey = 'dwarfs'

if survey == 'dwarfs':
    snap = '151'
elif survey == 'halos':
    snap = '137'

data_dir = '/home/rad/data/'+model+'/'+wind+'/'
output_dir = '/home/sapple/cgm/cos_samples/cos_'+survey+'/samples/'
output_file = output_dir + model+'_'+wind+'_'+snap+'.hdf5'


with h5py.File(output_dir+'particle_selection.h5', 'r') as f:
    plist = f['plist'][:]


snapfile = data_dir + 'snap_'+wind+'_'+snap+'.hdf5'
with h5py.File(snapfile, 'r') as in_file:
    header = in_file['Header']

with h5py.File(output_file, 'a') as out_file:
    if 'Header' not in out_file:
        out_file.copy(header, 'Header')
    for group in ['PartType0']:
        if group not in out_file:
            out_file.create_group(group)


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
            temp_dset = input_file[ptype][k][:]
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


