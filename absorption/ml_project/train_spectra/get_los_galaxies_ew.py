import numpy as np
import pandas as pd
import h5py
import sys
import yt
import caesar

import sys
sys.path.append('../make_spectra/')
from utils import read_h5_into_dict


def handle_periodic_boundaries(position, boxsize):
    position[position < 0] += boxsize
    position[position > boxsize] -= boxsize
    return position

def handle_periodic_mask(sample_gal_vpos, gal_vpos, vel_range, vel_boxsize):
    mask = (gal_vpos > sample_gal_vpos - vel_range) & (gal_vpos < sample_gal_vpos + vel_range)
    dv_right_edge = vel_boxsize - sample_gal_vpos
    if dv_right_edge < vel_range: 
        mask = mask | (gal_vpos < vel_range - dv_right_edge)
    dv_left_edge = sample_gal_vpos - vel_range
    if dv_left_edge < 0:
        mask = mask | (gal_vpos > vel_boxsize - np.abs(dv_left_edge))
    return mask

def move_edge_galaxies(los, gal_pos, rho, boxsize):
    for axis in range(2):
        if los[axis] < rho:
            right_gals = gal_pos[:, axis] > (boxsize - rho)
            gal_pos[:, axis][right_gals] -= boxsize
        elif los[axis] > (boxsize - rho):
            left_gals = gal_pos[:, axis] < rho
            gal_pos[:, axis][left_gals] += boxsize
    return gal_pos

def los_galaxies(los, gal_pos, rho, boxsize, vel_mask):
    new_gal_pos = move_edge_galaxies(los, gal_pos, rho, boxsize)
    dx = new_gal_pos[:, 0] - los[0]
    dy = new_gal_pos[:, 1] - los[1]
    dr = np.sqrt(dx**2 + dy**2)
    los_mask = dr < rho
    return np.arange(len(gal_pos))[los_mask*vel_mask]


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


def make_assoc_gal_df(gal_idx, gal_id, assoc_gal_id, main_gal, 
                      fr200_main_gal, fr200, orients, orient_idx, 
                      lines, lines_short, lines_dir,
                      model='m100n1024', wind='s50', snap='151'):

    gal_df = {}

    gal_df['associated_with'] = [gal_id]
    gal_df['gal_id'] = [assoc_gal_id]
    gal_df['main_gal'] = [False]

    gal_df['orient'] = [orients[orient_idx]]
    gal_df['fr200_main_gal'] = [fr200_main_gal]
    gal_df['fr200'] = [fr200]

    for line in lines:

        ew_dict = read_h5_into_dict(f'{lines_dir}{model}_{wind}_{snap}_ew_{line}.h5')

        gal_df[f'EW_{lines_short[lines.index(line)]}'] = \
                [ew_dict[f'ew_wave_{fr200_main_gal}r200'][gal_idx][orient_idx]]

    return pd.DataFrame(gal_df)


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

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    columns = ['EW_HI', 'EW_MgII', 'EW_CII', 'EW_SiIII', 'EW_CIV', 'EW_OVI',
               'associated_with', 'gal_id', 'main_gal', 'orient', 'fr200', 'fr200_main_gal']

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    lines_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
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

    gal_pos = np.array(gal_pos)
    # handle periodic boundaries
    gal_vpos = handle_periodic_boundaries(gal_vpos, vel_boxsize)
    sample_gal_vpos = handle_periodic_boundaries(sample_gal_vpos, vel_boxsize)

    ## get galaxies which may contribute to the LOS

    gal_ids_long = []
    associated_gal_ids = []
    all_fr200 = []
    all_orients = []
    all_actual_fr200 = []

    for i, gal in enumerate(sample_gal_ids):

        for j in range(len(fr200)):

            rho = sample_gal_r200[i] * fr200[j]
            vel_mask = handle_periodic_mask(sample_gal_vpos[i], gal_vpos, vel_range, vel_boxsize)

            los = np.tile(sample_gal_pos[i][:2], len(orients)).reshape(len(orients), 2)
            los[0][0] += rho # orient 0 degrees
            los[1][0] += (rho / sqrt2); los[1][1] += (rho / sqrt2) # orient 45 degrees 
            los[2][1] += rho # orient 90 degrees
            los[3][0] -= (rho / sqrt2); los[3][1] += (rho / sqrt2) # orient 135 degrees
            los[4][0] -= rho # orient 180 degrees
            los[5][0] -= (rho / sqrt2); los[5][1] -= (rho / sqrt2) # orient 225 degrees
            los[6][1] -= rho # orient 270 degrees
            los[7][0] += (rho / sqrt2); los[7][1] -= (rho / sqrt2) # orient 315 degrees
    
            for k in range(len(los)):
                ids = los_galaxies(los[k], gal_pos, rho, boxsize, vel_mask)
                
                if len(ids > 0):
                    ids = np.delete(ids, np.where(ids == gal)[0])
                
                if len(ids > 0):

                    x = gal_pos[ids][:, 0]
                    y = gal_pos[ids][:, 1]
                    dr = np.sqrt((x - los[k][0])**2. + (y - los[k][1])**2.)

                    gal_ids_long.extend(np.repeat(gal, len(ids)))
                    associated_gal_ids.extend(ids)
                    all_fr200.extend(np.repeat(fr200[j], len(ids)))
                    all_orients.extend(np.repeat(orients[k], len(ids)))
                    
                    all_actual_fr200.extend(dr / sample_gal_r200[i])

    gal_ids_long = np.array(gal_ids_long)
    associated_gal_ids = np.array(associated_gal_ids)
    all_fr200 = np.array(all_fr200)
    all_orients = np.array(all_orients)
    all_actual_fr200 = np.array(all_actual_fr200)

    ## match up the sample galaxies and associated galaxies with the LOS EWs

    gal_ids_idx = np.array([np.where(sample_gal_ids == i)[0][0] for i in gal_ids_long]).flatten()
    fr200_idx = np.array([np.where(fr200 == i)[0][0] for i in all_fr200]).flatten()
    orient_idx = np.array([np.where(orients == i)[0][0] for i in all_orients]).flatten()

    data = pd.DataFrame(columns=columns)

    for i, gal in enumerate(sample_gal_ids):

        dataframes = [data]

        dataframes.append(make_main_gal_df(i, gal, fr200, norients, lines, lines_short, lines_dir))

        where = np.where(gal_ids_idx == i)[0] 
        
        if len(where) > 0.:
            for j in range(len(where)):
                dataframes.append(make_assoc_gal_df(i, gal, associated_gal_ids[where[j]], False, 
                                                    fr200[fr200_idx[where[j]]], all_actual_fr200[where[j]], orients, orient_idx[where[j]], 
                                                    lines, lines_short, lines_dir ))

        data = pd.concat(dataframes, ignore_index=True)

    ## get additional galaxy properties

    data['mass'] = gal_sm[np.array(data['gal_id']).astype(int)] 
    data['ssfr'] = gal_ssfr[np.array(data['gal_id']).astype(int)]
    data['kappa_rot'] = gal_kappa_rot[np.array(data['gal_id']).astype(int)]

    split = 0.8
    train = np.random.rand(len(df_full)) < split
    df_full['train_mask'] = train

    data.to_csv(f'data/{model}_{wind}_{snap}_ew.csv')
