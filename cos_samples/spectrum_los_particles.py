import pygad as pg
import numpy as np
import h5py
import yt
from yt.units.yt_array import YTArray, YTQuantity
from generate_spectra import generate_pygad_spectrum
import matplotlib.pyplot as plt

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

line = 'H1215'
lambda_rest = 1215.


los = np.array([15092.04743875, 82958.41646495]) # los in kpc/h
gal = 11486

sample_file = 'cos_dwarfs/samples/m100n1024_s50_cos_dwarfs_sample.h5'
with h5py.File(sample_file, 'r') as f:
    vpos = f['vgal_position'][0][2]

snapfile = '/home/sapple/cgm/cos_samples/cos_dwarfs/samples/m100n1024_s50_151.hdf5'
#snapfile = '/home/rad/data/m100n1024/s50j7k/snap_m100n1024_151.hdf5'

# Load in snapshot for pygad spectra generation:
s = pg.Snapshot(snapfile)


# Get some info from yt:
ds = yt.load(snapfile)
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(ds.current_redshift).in_units('km/s/kpc')
vbox = ds.domain_right_edge[2].in_units('kpc') * hubble / ds.hubble_constant / (1.+ds.current_redshift)

# Set some spectrum parameters:
snr = 12.
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
plt.axvline(vpos, lw=1, c='k')
plt.axvline(vpos + vel_range , lw=1, c='k', ls='--')
plt.axvline(vpos - vel_range , lw=1, c='k', ls='--')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Flux')
plt.savefig('test_snapshot.png')
plt.clf()

