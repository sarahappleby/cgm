import matplotlib.pyplot as plt
import h5py
import os
import sys
import numpy as np 
from plotting_methods import get_cb_colours, read_phase_stats

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
palette_name = 'tol'

alpha = .8
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex

snap = '151'
model = sys.argv[1]
wind = sys.argv[2]

data_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'_'+snap+'/'
savedir = '/home/sapple/cgm/budgets/plots/'

plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
                'Wind', 'Dust', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$', 
                      r'Cool CGM $(T < T_{\rm photo})$', 'Wind', 'Dust', 'ISM', 'Stars']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

metal_stats_file = data_dir+model+'_'+wind+'_'+snap+'_metal_budget_stats.h5'
frac_stats_file = data_dir+model+'_'+wind+'_'+snap+'_avail_metal_frac_stats.h5'

if os.path.isfile(frac_stats_file):
    frac_stats = read_phase_stats(frac_stats_file, plot_phases, stats)
else:
	print('Run plot_mass_frac_available first!')
	quit()

if os.path.isfile(metal_stats_file):
    metal_stats = read_phase_stats(metal_stats_file, plot_phases, stats)
else:
	print('Run plot_mass_actual first!')
	quit()

fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey='row', sharex='col')
ax = ax.flatten()

# Plot actual metal masses:

for i, phase in enumerate(plot_phases):
    ax[0].errorbar(metal_stats['smass_bins'], metal_stats['all'][phase]['median'], yerr=metal_stats['all'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[1].errorbar(metal_stats['smass_bins'], metal_stats['star_forming'][phase]['median'], yerr=metal_stats['star_forming'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[2].errorbar(metal_stats['smass_bins'], metal_stats['quenched'][phase]['median'], yerr=metal_stats['quenched'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])

ann_labels = ['All', 'Star forming', 'Quenched']
ann_x = [0.88, 0.63, 0.7]
for i in range(3):
    ax[i].annotate(ann_labels[i], xy=(ann_x[i], 0.05), xycoords='axes fraction',size=18,
            bbox=dict(boxstyle='round', fc='white'))

for i in range(3):
    ax[i].set_xlim(min_mass, metal_stats['smass_bins'][-1]+0.5*dm)
    ax[i].set_ylim(5.5, 11.5)
    #ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[0].set_ylabel(r'$\textrm{log} (M_Z / \textrm{M}_{\odot})$')
ax[0].legend(loc=2, fontsize=13, framealpha=0.)

# Plot metal mass fractions:

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['all'][phase]['median']
    
running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    ax[3].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['all'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['all'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['star_forming'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    ax[4].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['star_forming'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['star_forming'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['quenched'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    ax[5].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['quenched'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['quenched'][phase]['median'] / total

for i in range(3, 6):
    ax[i].set_xlim(frac_stats['smass_bins'][0], frac_stats['smass_bins'][-1])
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[3].set_ylabel(r'$f_{Z\ {\rm Total}}$')
ax[3].legend(loc=1, fontsize=13)
fig.subplots_adjust(wspace=0., hspace=0.)

plt.savefig(savedir+model+'_'+wind+'_'+snap+'_metal_budget.png', bbox_inches = 'tight')
plt.clf()
