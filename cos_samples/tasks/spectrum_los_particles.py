import pygad as pg
import numpy as np
import h5py
import yt
from yt.units.yt_array import YTArray, YTQuantity
from generate_spectra import generate_pygad_spectrum
import matplotlib.pyplot as plt
from pygadgetreader import readsnap
from save_new_dataset import *
import os

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

line = 'H1215'
lambda_rest = 1215.
los = np.array([25000, 25000]) # los in kpc/h
model = 'm50n512'
wind = 's50'
snap = '151'
sample_dir = './los_particles_test/'
output_snap = sample_dir+'m50n512_s50_151.hdf5'

data_dir = '/home/rad/data/'+model+'/'+wind+'/'
snapfile = data_dir + 'snap_'+model+'_'+snap +'.hdf5'

if not os.path.isfile(output_snap):

    hsml = readsnap(snapfile, 'SmoothingLength', 'gas', suppress=1, units=1)  # in kpc/h
    gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) # in kpc/h

    x_dist = np.abs(los[0] - gas_pos[:, 0])
    y_dist = np.abs(los[1] - gas_pos[:, 1])
    dist_mask = (x_dist < hsml) & (y_dist < hsml)
    plist = np.arange(len(hsml))[dist_mask]

    plist = np.unique(np.sort(plist))
    numpart = np.zeros(6, dtype='int')
    numpart[0] = len(plist)

    prepare_out_file(snapfile, output_snap, numpart)
    make_new_dataset(snapfile, output_snap, plist, 2)

snapshots = [snapfile, output_snap]
names = ['old', 'new']

for i, snapshot in enumerate(snapshots):
    # Load in snapshot for pygad spectra generation:
    s = pg.Snapshot(snapshot)


    # Get some info from yt:
    ds = yt.load(snapshot)
    co = yt.utilities.cosmology.Cosmology()
    hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
    vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)

    # Set some spectrum parameters:
    periodic_vel = True
    vel_range = 600. # km/s
    sigma_vel = 6. # km/s
    Nbins = int(np.rint(vbox / sigma_vel))
    v_limits = [-600., vbox.value+600.]

    taus, col_densities, dens, temps, metal_frac, vel, v_edges, restr_column = \
            pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line, v_limits, Nbins=Nbins, return_los_phys=True)

    fluxes = np.exp(-1.*taus)
    velocities = 0.5 * (v_edges[1:] + v_edges[:-1])
    wavelengths = vel_to_wave(velocities, lambda_rest, np.array(pg.physics.cosmology.c.in_units_of('km/s')), s.redshift)

    plt.plot(velocities, fluxes)
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux')
    plt.savefig('test_snapshot_'+names[i]+'.png')
    plt.clf()

