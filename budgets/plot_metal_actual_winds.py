import matplotlib.pyplot as plt
import h5py
import os
import sys
import caesar
import numpy as np 
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
palette_name = 'tol'

alpha = 1.
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex

snap = '151'
winds = ['s50', 's50nox', 's50nojet', 's50noagn']
model = 'm50n512'
boxsize = 50000.

system = sys.argv[1]
if system == 'laptop':
	savedir = '/home/sarah/cgm/budgets/plots/'
elif system == 'ursa':
	savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
			  'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
			  'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
				'Wind', 'Dust', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$', 
					  r'Cool CGM $(T < T_{\rm photo})$', 'Wind', 'Dust', 'ISM', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()

for w, wind in enumerate(winds):

	if system == 'laptop':
		data_dir = '/home/sarah/cgm/budgets/data/'+model+'_'+wind+'/'
	elif system == 'ursa':
		data_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'/'
	metal_stats_file = data_dir+model+'_'+wind+'_'+snap+'_metal_budget_stats.h5'

	if os.path.isfile(metal_stats_file):

		metal_stats = read_phase_stats(metal_stats_file, plot_phases, stats)

	else:

		# get the galaxy data:
		#caesarfile = '/home/sarah/data/caesar_snap_m12.5n128_135.hdf5'
		#sim = caesar.load(caesarfile)
		caesarfile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
		sim = caesar.quick_load(caesarfile)
		quench = -1.8  + 0.3*sim.simulation.redshift
		central = np.array([i.central for i in sim.galaxies])
		gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies])[central]
		gal_sfr = np.array([i.sfr.in_units('Msun/Gyr') for i in sim.galaxies])[central]
		gal_ssfr = np.log10(gal_sfr / gal_sm) 

		gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies])[central]

		metal_budget = read_phases(data_dir+'metal_budget.h5', all_phases)

		metal_stats = {}
		mass_bins = get_bin_edges(min_mass, max_mass, dm)
		metal_stats['smass_bins'] = get_bin_middle(np.append(mass_bins, mass_bins[-1] + dm)) 

		mask = np.array([True] * len(gal_sm))
		metal_stats['all'] = get_phase_stats(gal_sm, gal_pos, metal_budget, mask, all_phases, mass_bins, boxsize, logresults=True)

		mask = gal_ssfr > quench
		metal_stats['star_forming'] = get_phase_stats(gal_sm, gal_pos, metal_budget, mask, all_phases, mass_bins, boxsize, logresults=True)
		
		mask = gal_ssfr < quench
		metal_stats['quenched'] = get_phase_stats(gal_sm, gal_pos, metal_budget, mask, all_phases, mass_bins, boxsize, logresults=True)

		write_phase_stats(metal_stats_file, metal_stats, all_phases, stats)

	for i, phase in enumerate(plot_phases):
		ax[w].errorbar(metal_stats['smass_bins'], metal_stats['all'][phase]['median'], yerr=metal_stats['all'][phase]['percentile_25_75'], 
					capsize=3, color=colours[i], label=plot_phases_labels[i])

	ax[w].set_xlim(min_mass, metal_stats['smass_bins'][-1]+0.5*dm)
	ax[w].set_ylim(5.5, 11.5)
	ax[w].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
	ax[w].set_ylabel(r'$\textrm{log} (M_Z / \textrm{M}_{\odot})$')
	ax[w].annotate(wind, xy=(0.05, 0.05), xycoords='axes fraction')

ax[0].legend(loc=2, fontsize=11, framealpha=0.)
plt.savefig(savedir+model+'_'+snap+'_metal_actual_winds.png', bbox_inches = 'tight')
plt.clf()
