# Script to identify LOS particles for a given selected galaxy, for LOS parallel to the z axis of the simulation
# Run using sub_los_particles.sh

# for each galaxy in the sample, identify particles that contribute towards the line of sight using the smoothing length of each particle
# save new dataset containing only these particles using Chris's approach

from pygadgetreader import readsnap
import numpy as np
import pygad as pg
from numba import njit
import h5py
import os
import gc
import sys

@njit
def get_los_particles(los, gas_pos, hsml, wind_mask):
    x_dist = np.abs(los[0] - gas_pos[:, 0])
    y_dist = np.abs(los[1] - gas_pos[:, 1])
    hyp_sq = x_dist**2 + y_dist**2
    dist_mask = hyp_sq < hsml**2
    partids_los = np.arange(len(hsml))[dist_mask * wind_mask]
    return partids_los

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':
    
    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    sample_gal = int(sys.argv[4]) # supply the gal id that we want from command lin

    sqrt2 = np.sqrt(2.)
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    data_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    snapfile = f'{data_dir}{model}_{wind}_{snap}.hdf5'
    
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    h = s.cosmology.h_0
    quench = quench_thresh(redshift)

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as f:
        gal_id = f['gal_ids'][:].astype('int')[sample_gal]
        pos = f['position'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving
        r200 = f['halo_r200'][:][sample_gal] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving
        ssfr = f['ssfr'][:][sample_gal]

    if ssfr > quench:
        plist_file = f'{sample_dir}{model}_{wind}_{snap}_particle_selection_star_forming.h5'
    elif (ssfr < quench) & (ssfr > quench -1):
        plist_file = f'{sample_dir}{model}_{wind}_{snap}_particle_selection_green_valley.h5'
    elif ssfr == -14:
        plist_file = f'{sample_dir}{model}_{wind}_{snap}_particle_selection_quenched.h5'

    if os.path.isfile(plist_file):
        with h5py.File(plist_file, 'r') as hf:
            if f'plist_{gal_id}' in hf.keys():
                sys.exit()

    hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving
    gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
    wind_mask = gas_delaytime == 0.

    partids = np.array([])

    for i in range(nbins_fr200):
        los = np.array([pos[:2].copy(), ]*8)
        rho = r200 * fr200[i]
        los[0][0] += rho
        los[1][0] += (rho / sqrt2); los[1][1] += (rho / sqrt2)
        los[2][1] += rho
        los[3][0] -= (rho / sqrt2); los[3][1] += (rho / sqrt2)
        los[4][0] -= rho
        los[5][0] -= (rho / sqrt2); los[5][1] -= (rho / sqrt2)
        los[6][1] -= rho
        los[7][0] += (rho / sqrt2); los[7][1] -= (rho / sqrt2)

        for l in los:
            partids_los = get_los_particles(l, gas_pos, hsml, wind_mask)
            partids = np.append(partids, partids_los)
            del partids_los

    partids = np.unique(np.sort(partids))
    with h5py.File(plist_file, 'a') as f:
        f.create_dataset(f'plist_{gal_id}', data=np.array(partids))
    del partids

