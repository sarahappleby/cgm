# Script for creating subsets of particles from gizmo sims
# Based on scripts by Sydney Lower and Chris Lovell
# https://gist.github.com/christopherlovell/5a504c2c9d26efb6e073324d80c755a6 



import h5py
import caesar
import numpy as np

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

    model = 'm25n512'
    wind = 's50'
    survey = 'dwarfs'
    verbose = 2

    if survey == 'dwarfs':
        snap = '151'
    elif survey == 'halos':
        snap = '137'

    if (model == 'm50n512') & (survey == 'halos'):
        ignore_cos_gals = [18, 29]
    if (model == 'm25n512') & (survey == 'dwarfs'):
        ignore_cos_gals = [10, 17, 36]
    if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
        ignore_simba_gals = [list(range(num*5, (num+1)*5)) for num in ignore_cos_gals]
        ignore_simba_gals = [item for sublist in ignore_simba_gals for item in sublist]

    data_dir = '/home/rad/data/'+model+'/'+wind+'/'
    snapfile = data_dir+'snap_'+model+'_'+snap+'.hdf5'

    output_dir = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'
    output_file = output_dir + model+'_'+wind+'_'+snap+'.hdf5'

    sample_file = model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(sample_file, 'r') as f:
        gal_ids = np.array(f['gal_ids'][:], dtype='int')

    plist = np.array([])
    with h5py.File(output_dir+wind+'_particle_selection.h5', 'r') as f:
        for i, gal in enumerate(gal_ids):
            if (model == 'm50n512') & (survey == 'halos') & (i in ignore_simba_gals):
                print('Ignoring certain m50n512 COS-Halos galaxies')
                continue
            elif (model == 'm25n512') & (survey == 'dwarfs') & (i in ignore_simba_gals):
                print('Ignoring certain m25n512 COS-Dwarfs galaxies')
                continue
            else:
                print('Reading in galaxy '+str(i))
                plist = np.append(plist, np.array(f['plist_'+str(i)+'_'+str(gal)][:], dtype='int'))

    plist = np.unique(np.sort(plist)).astype('int')

    numpart = np.zeros(6, dtype='int')
    numpart[0] = len(plist)

    prepare_out_file(snapfile, output_file, numpart)

    make_new_dataset(snapfile, output_file, plist, verbose)
