# Given some galaxy selection in the main m50n512 Simba box, this script will find these matching galaxies in the other wind boxes.


import h5py
import numpy as np
import caesar
import yt
from ignore_gals import * 

def check_halo_sample(prog_index, obj1, obj2, gal_id):
    gal = obj1.galaxies[gal_id]
    halo1 = gal.parent_halo_index
    halo2 = obj2.halos[prog_index[halo1]]
    return halo2.central_galaxy.GroupID

if __name__ == '__main__':

    model = 'm50n512'
    wind1 = 's50j7k'
    wind_options = ['s50nojet', 's50nofb']
    snap = '137'
    survey = 'halos'

    # ignore these as the main s50 sample has insufficient galaxies
    ignore_simba_gals, ngals_each = get_ignore_simba_gals(model, survey)

    sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind1+'_cos_'+survey+'_sample.h5'
    with h5py.File(sample_file, 'r') as f:
        gal_ids = np.array(f['gal_ids'][:], dtype='int')
        cos_ids = f['cos_ids'][:]

    infile = '/home/rad/data/'+model+'/'+wind1+'/Groups/'+model+'_'+snap+'.hdf5'
    obj1 = caesar.load(infile)

    for wind2 in wind_options:

        match_file = './m50n512/match_halos_'+snap+'.hdf5'
        with h5py.File(match_file, 'r') as f:
            prog_index = f[wind1+'_'+wind2][:]

        infile = '/home/rad/data/'+model+'/'+wind2+'/Groups/'+model+'_'+snap+'.hdf5'
        obj2 = caesar.load(infile)

        new_gal_ids = np.array([np.nan] * len(gal_ids))

        for i, gal_id in enumerate(gal_ids):
            if not i in ignore_simba_gals:
                new_gal_ids[i] = check_halo_sample(prog_index, obj1, obj2, gal_id)

        co = yt.utilities.cosmology.Cosmology()
        hubble = co.hubble_parameter(obj2.simulation.redshift).in_units('km/s/kpc')
        redshift = obj2.simulation.redshift

        gal_sm = yt.YTArray([i.masses['stellar'].in_units('Msun') for i in obj2.galaxies], 'Msun')
        gal_sfr = yt.YTArray([i.sfr.in_units('Msun/yr') for i in obj2.galaxies], 'Msun/yr')
        gal_ssfr = gal_sfr / gal_sm
        gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
        gal_pos = yt.YTArray([i.pos.in_units('kpc/h') for i in obj2.galaxies], 'kpc/h')
        gal_vels = yt.YTArray([i.vel.in_units('km/s') for i in obj2.galaxies], 'km/s')
        gal_sm = np.log10(gal_sm)
        gal_recession = gal_pos.in_units('kpc')*hubble
        gal_vgal_pos = gal_vels + gal_recession
        gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in obj2.galaxies ])

        new_mass = np.array([np.nan] * len(new_gal_ids))
        new_ssfr = np.array([np.nan] * len(new_gal_ids))
        new_gas_frac = np.array([np.nan] * len(new_gal_ids))
        new_pos = np.transpose([[np.nan] * len(new_gal_ids) for _ in range(3)])
        new_vgal_pos = np.transpose([[np.nan] * len(new_gal_ids) for _ in range(3)])

        new_mass[~np.isnan(new_gal_ids)] = gal_sm[new_gal_ids[~np.isnan(new_gal_ids)].astype(int)]
        new_ssfr[~np.isnan(new_gal_ids)] = gal_ssfr[new_gal_ids[~np.isnan(new_gal_ids)].astype(int)]
        new_gas_frac[~np.isnan(new_gal_ids)] = gal_gas_frac[new_gal_ids[~np.isnan(new_gal_ids)].astype(int)]
        new_pos[~np.isnan(new_gal_ids)] = gal_pos[new_gal_ids[~np.isnan(new_gal_ids)].astype(int)]
        new_vgal_pos[~np.isnan(new_gal_ids)] = gal_vgal_pos[new_gal_ids[~np.isnan(new_gal_ids)].astype(int)]

        new_sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind2+'_cos_'+survey+'_sample.h5'
        with h5py.File(new_sample_file, 'a') as f:
            f.create_dataset('gal_ids', data=np.array(new_gal_ids))
            f.create_dataset('cos_ids', data=np.array(cos_ids))
            f.create_dataset('mass', data=np.array(new_mass))
            f.create_dataset('ssfr', data=np.array(new_ssfr))
            f.create_dataset('gas_frac', data=np.array(new_gas_frac))
            f.create_dataset('position', data=np.array(new_pos))
            f.create_dataset('vgal_position', data=np.array(new_vgal_pos))
            f.attrs['pos_units'] = 'kpc/h'
            f.attrs['mass_units'] = 'Msun'
            f.attrs['ssfr_units'] = 'Msun/yr'
            f.attrs['vel_units'] = 'km/s'
