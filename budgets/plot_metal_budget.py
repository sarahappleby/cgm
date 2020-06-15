import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
from plotting_methods import *

min_mass = 9.5
max_mass = 12.
dm = 0.5 # dex

# if model == 'm100n1024':
#     boxsize = 100000.
# elif model == 'm50n512':
#     boxsize = 50000.

boxsize = 12500.
metaldata_dir = '/home/sarah/cgm/budgets/data/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)',
			  'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
			  'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)', 
			  'ISM', 'Wind', 'Dust', 'Stars']
colours = ['m', 'tab:orange', 'g', 'b', 'c', 'tab:pink', 'r']
stats = ['median', 'cosmic_std', 'ngals']

metal_stats_file = metaldata_dir+'metal_budget_stats.h5'

if os.path.isfile(metal_stats_file):

	metal_stats = {p: {} for p in phases}
	with h5py.File(metal_stats_file, 'r') as hf:
		for phase in plot_phases:
			for stat in stats:
				metal_stats[phase][stat] = hf[phase][stat][:]

		plot_bins = hf['smass_bins'][:]

else:

	mass_bins = get_bin_edges(min_mass, max_mass, dm)
	plot_bins = get_bin_middle(mass_bins)

	# get the galaxy data:
	caesarfile = '/home/sarah/data/caesar_snap_m12.5n128_135.hdf5'
	sim = caesar.load(caesarfile)
	central = np.array([i.central for i in sim.galaxies])
	gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies])[central]
	gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies])[central]

	# get the metal budget data:
	metal_budget = {}
	with h5py.File(metaldata_dir+'metal_budget.h5', 'r') as hf:
		for phase in all_phases:
			metal_budget[phase] = hf[phase][:]

	metal_stats = {phase: {} for phase in all_phases}
	binned_pos = bin_data(gal_sm, gal_pos, 10.**mass_bins)
	for phase in all_phases:
		binned_data = bin_data(gal_sm, metal_budget[phase], 10.**mass_bins)
		
		medians = np.zeros(len(plot_bins))
		cosmic_stds = np.zeros(len(plot_bins))
		for i in range(len(plot_bins)):
			medians[i], cosmic_stds[i] = get_cosmic_variance(binned_data[i], binned_pos[i], boxsize)

		metal_stats[phase]['median'], metal_stats[phase]['cosmic_std'] = convert_to_log(medians, cosmic_stds)
		metal_stats[phase]['ngals'] = [len(j) for j in binned_data]

	with h5py.File(metal_stats_file, 'a') as hf:
		for phase in all_phases:
			grp = hf.create_group(phase)
			for stat in stats:
				grp.create_dataset(stat, data=np.array(metal_stats[phase][stat]))

		hf.create_dataset('smass_bins', data=np.array(plot_bins))

for i, phase in enumerate(plot_phases):
	plt.errorbar(plot_bins, metal_stats[phase]['median'], yerr=metal_stats[phase]['cosmic_std'], color=colours[i], label=phase)

plt.legend(loc=2)
plt.xlabel('M*')
plt.ylabel('Metal Mass')
plt.xlim(min_mass, max_mass)
plt.savefig('/home/sarah/cgm/budgets/plots/metal_mufasa_128.png')
plt.clf()