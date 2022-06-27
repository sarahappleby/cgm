# Get the typical rhalf of the galaxies in a mass range

import caesar
import h5py
import numpy as np

model = 'm100n1024'
wind = 's50'
snap = '151'

data_dir = f'/home/rad/data/{model}/{wind}/Groups/'
sim = caesar.load(f'{data_dir}{model}_{snap}.hdf5')

delta_m = 0.25
min_m = 9.
nbins_m = 10.
mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)

mass_plot_titles = []
for i in range(nbins_m):
    mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i]+delta_m}')


sat_mask = ~np.array([i.central for i in sim.galaxies])
sat_ids = np.arange(len(sim.galaxies))[sat_mask]

h = sim.simulation.hubble_constant
redshift = sim.simulation.redshift

rhalf = np.array([sim.galaxies[i].radii['stellar_half_mass'].in_units('kpc/h') for i in sat_ids]) * (1+redshift)
mass = np.log10([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in sat_ids])

median_rhalf = np.zeros(len(mass_bins))
ngals = np.zeros(len(mass_bins))
for i in range(len(mass_bins)):
    mass_mask = (mass > mass_bins[i]) & (mass < mass_bins[i]+ delta_m)
    median_rhalf[i] = np.around(np.nanmedian(rhalf[mass_mask]), 2)
    ngals[i] = len(rhalf[mass_mask])

print(mass_bins)
print(median_rhalf_all)
