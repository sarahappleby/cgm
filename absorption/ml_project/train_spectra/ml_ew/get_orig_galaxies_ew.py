import numpy as np
import pandas as pd
import h5py
import sys
import yt
import caesar

import sys
sys.path.append('../make_spectra/')
from utils import read_h5_into_dict


def make_main_gal_df(gal_idx, gal_id, fr200, norients, lines, lines_short, lines_dir,
                model='m100n1024', wind='s50', snap='151'):

    gal_df = pd.DataFrame()

    gal_df['associated_with'] = np.repeat(gal_id, norients*len(fr200))
    gal_df['gal_id'] = np.repeat(gal_id, norients*len(fr200))
    gal_df['main_gal'] = np.repeat(True, norients*len(fr200)).astype(bool)

    gal_df['orient'] = np.tile(orients, len(fr200))
    gal_df['fr200'] = np.repeat(fr200, norients)
    gal_df['fr200_main_gal'] = np.repeat(fr200, norients)

    for line in lines:

        ew_dict = read_h5_into_dict(f'{lines_dir}{model}_{wind}_{snap}_ew_{line}.h5')

        gal_df[f'EW_{lines_short[lines.index(line)]}'] = np.zeros(norients*len(fr200))

        for j in range(len(fr200)):

            gal_df[f'EW_{lines_short[lines.index(line)]}'][j*norients:(j+1)*norients] = \
                    ew_dict[f'ew_wave_{fr200[j]}r200'][gal_idx]

    return gal_df



if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sqrt2 = np.sqrt(2.)
    vel_range = 600. # km/s
    vel_boxsize = 10000. #km/s
    boxsize = 100000. # kpc/h
    orients = [0, 45, 90, 135, 180, 225, 270, 315]
    norients = 8
    mlim = 9.

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    columns = ['EW_HI', 'EW_MgII', 'EW_CII', 'EW_SiIII', 'EW_CIV', 'EW_OVI',
               'associated_with', 'gal_id', 'main_gal', 'orient', 'fr200', 'fr200_main_gal']

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    sample_dir = f'/disk04/sapple/data/samples/'
    lines_dir = f'/disk04/sapple/data/normal/results/'
    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    co = yt.utilities.cosmology.Cosmology(hubble_constant=sim.simulation.hubble_constant,
                                          omega_matter=sim.simulation.omega_matter,
                                          omega_lambda=sim.simulation.omega_lambda)
    hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
    redshift = sim.simulation.redshift

    gal_cent = np.array([i.central for i in sim.galaxies])
    gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
    gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
    gal_ssfr = gal_sfr / gal_sm
    gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
    gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
    gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
    gal_sm = np.log10(gal_sm)
    gal_recession = gal_pos.in_units('kpc')*hubble
    gal_vpos = gal_vels + gal_recession
    gal_vpos = np.array(gal_vpos[:, 2])
    gal_kappa_rot = np.array([i.rotation['total_kappa_rot'].in_units('dimensionless') for i in sim.galaxies])

    ## get the original galaxy sample

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        sample_gal_ids = sf['gal_ids'][:]
        sample_gal_pos = (sf['position'][:]  *  (1.+redshift))
        sample_gal_vpos = sf['vgal_position'][:][:, 2]
        sample_gal_r200 = sf['halo_r200'][:] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving

    data = pd.DataFrame(columns=columns)

    for i, gal in enumerate(sample_gal_ids):

        dataframes = [data]

        dataframes.append(make_main_gal_df(i, gal, fr200, norients, lines, lines_short, lines_dir))

        data = pd.concat(dataframes, ignore_index=True)

    ## get additional galaxy properties

    data['mass'] = gal_sm[np.array(data['gal_id']).astype(int)] 
    data['ssfr'] = gal_ssfr[np.array(data['gal_id']).astype(int)]
    data['kappa_rot'] = gal_kappa_rot[np.array(data['gal_id']).astype(int)]

    mass_mask = data['mass'] > mlim

    data = data[mass_mask]

    split = 0.8
    train = np.random.rand(len(data)) < split
    data['train_mask'] = train

    data.to_csv(f'data/{model}_{wind}_{snap}_ew_orig_centrals.csv')
