# Incomplete; at one point I wanted to compute the angular momentum of each sample galaxy from their halo particles.

import numba
import numpy as np
import h5py
import sys
from pygadgetreader import readsnap

@jit(nopython=True)
def compute_L(pos, vel, mass):
    L_all = np.cross(pos, vel) * np.repeat(mass, 3).reshape(len(mass), 3)
    return np.sum(L_all, axis=0)

@jit(nopython=True)
def compute_orientation(L, vector=np.array([0, 0, 1]))
    L_norm = L / np.linalg(L)
    vec_norm = vector / np.linalg(vector)
    cosine = np.dot(L_norm, vec_norm)
    angle = np.arccos(np.clip(c, -1, 1))
    return angle

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

# Read in the galaxy sample

data_dir = f'/home/rad/data/{model}/{wind}/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift
snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'

gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) / (hubble.value*(1.+redshift)) # in kpc
gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / hubble.value # in Mo
gas_vels = readsnap(snapfile, 'vel', 'gas', suppress=1, units=0) # in km/s
star_pos = readsnap(snapfile, 'pos', 'star', suppress=1, units=1) / (hubble.value*(1.+redshift)) # in kpc
star_mass = readsnap(snapfile, 'mass', 'star', suppress=1, units=1) / hubble.value # in Mo
star_vels = readsnap(snapfile, 'vel', 'star', suppress=1, units=0) # in km/s
bh_pos = readsnap(snapfile, 'pos', 'bndry', suppress=1, units=1) / (hubble.value*(1.+redshift)) # in kpc

for i in range(len(gal_ids)):

    # get the gas and star particles for computing the angular momentum
    pos -= bh_pos[sim.galaxies[i].bhlist[0]]

    # Compute angular momentum and orientation angle

