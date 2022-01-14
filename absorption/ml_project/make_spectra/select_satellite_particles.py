"""
1) Start from the snap file which contains only particles along the line of sight
2) Read in the caesar list of satellite galaxies and their positions and sizes
3) For each satellite, get the particles in the snap file which are associated with the satellite
4) Get unique particle ids, save out particle ids file

"""
import sys
import numpy as np
import h5py
import caesar
from pygadgetreader import readsnap


def get_particle_distance(gal_pos, gas_pos, gas_hsml):
    dist = np.linalg.norm(np.abs(gal_pos - gas_pos), axis=1)
    dist -= gas_hsml
    return dist


def find_satellite_particles(gal_pos, gal_rad, gas_dist):
    sat_mask = gas_dist < gal_rad
    return np.arange(len(gas_dist))[sat_mask]


if __name__ == '__main__':
    
    model = 'm100n1024'
    snap = '151'
    wind = 's50'
    num = int(sys.argv[1])

    log_frad_min = 0.
    log_frad_max = 3.
    log_dfrad = 0.5
    log_frad = np.arange(log_frad_min, log_frad_max+log_dfrad, log_dfrad)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    snapfile = f'{sample_dir}{model}_{wind}_{snap}.hdf5'
   
    gas_hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h, comoving
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h, comoving    

    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

    h = sim.simulation.hubble_constant
    redshift = sim.simulation.redshift

    sat_mask = ~np.array([i.central for i in sim.galaxies])
    sat_ids = np.arange(len(sim.galaxies))[sat_mask]

    partids = {}
    for lf in log_frad: 
        partids[f'log_frad_{lf}'] = np.array([])

    i = sat_ids[num]

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_satellite_only_{lf}log_frad.h5', 'a') as f:
        if f'plist_{i}' in f.keys():
            quit()

    print(f'Doing galaxy {num}/{len(sat_ids)}')

    gal = sim.galaxies[i]
    pos = np.array(gal.pos.in_units('kpc/h')) * (1+redshift)
    rad = np.array(gal.radii['stellar_half_mass'].in_units('kpc/h')) * (1+redshift)

    gas_dist = get_particle_distance(pos, gas_pos, gas_hsml)
        
    for lf in log_frad:
        frad = 10**lf
        partids[f'log_frad_{lf}'] = find_satellite_particles(pos, rad*frad, gas_dist)
        partids[f'log_frad_{lf}'] = np.unique(np.sort(partids[f'log_frad_{lf}']))
        with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_satellite_only_{lf}log_frad.h5', 'a') as f:
            if not f'plist_{i}' in f.keys():
                f.create_dataset(f'plist_{i}', data=np.array(partids[f'log_frad_{lf}']))
