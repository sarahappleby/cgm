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

mass_stats_file = data_dir+model+'_'+wind+'_'+snap+'_mass_budget_stats.h5'
frac_stats_file = data_dir+model+'_'+wind+'_'+snap+'_avail_frac_stats.h5'

if os.path.isfile(frac_stats_file):
    frac_stats = read_phase_stats(frac_stats_file, plot_phases, stats)
else:
	print('Run plot_mass_frac_available first!')
	quit()

if os.path.isfile(mass_stats_file):
    mass_stats = read_phase_stats(mass_stats_file, plot_phases, stats)
else:
	print('Run plot_mass_actual first!')
	quit()

fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey='row', sharex='col')
ax = ax.flatten()

# Plot absolute masses

for i, phase in enumerate(plot_phases):
    ax[0].errorbar(mass_stats['smass_bins'], mass_stats['all'][phase]['median'], yerr=mass_stats['all'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[1].errorbar(mass_stats['smass_bins'], mass_stats['star_forming'][phase]['median'], yerr=mass_stats['star_forming'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[2].errorbar(mass_stats['smass_bins'], mass_stats['quenched'][phase]['median'], yerr=mass_stats['quenched'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])

ann_labels = ['All', 'Star forming', 'Quenched']
ann_x = [0.88, 0.64, 0.71]
for i in range(3):
    ax[i].annotate(ann_labels[i], xy=(ann_x[i], 0.05), xycoords='axes fraction',size=16,
            bbox=dict(boxstyle='round', fc='white'))

for i in range(3):
    ax[i].set_xlim(min_mass, mass_stats['smass_bins'][-1]+0.5*dm)
    ax[i].set_ylim(6.5, 14.5)
    #ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[0].set_ylabel(r'$\textrm{log} (M / \textrm{M}_{\odot})$')
ax[0].legend(loc=2, fontsize=13, framealpha=0.)

# Plot mass fractions

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['all'][phase]['median']
    
running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue
    ax[3].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['all'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['all'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['star_forming'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue
    ax[4].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['star_forming'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['star_forming'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['quenched'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue    
    ax[5].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['quenched'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['quenched'][phase]['median'] / total

for i in range(3, 6):
    ax[i].set_xlim(frac_stats['smass_bins'][0], frac_stats['smass_bins'][-1])
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[3].set_ylabel(r'$f_{\rm Total}$')
ax[5].legend(loc=2, fontsize=13)

fig.subplots_adjust(wspace=0., hspace=0.)
plt.savefig(savedir+model+'_'+wind+'_'+snap+'_mass_budget_avail.png', bbox_inches = 'tight')
plt.clf()
