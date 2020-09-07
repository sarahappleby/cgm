import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
import sys
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

alpha = .9
palette_name = 'tol'
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex

snap = '151'
model = sys.argv[1]
wind = sys.argv[2]

if model == 'm100n1024':
    boxsize = 100000.
elif model == 'm50n512':
    boxsize = 50000.

fracdata_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'/'
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

frac_stats_file = fracdata_dir+model+'_'+wind+'_'+snap+'_avail_frac_stats.h5'

if os.path.isfile(frac_stats_file):

    frac_stats = read_phase_stats(frac_stats_file, plot_phases, stats)

else:

    caesarfile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
    sim = caesar.quick_load(caesarfile)
    quench = -1.8  + 0.3*sim.simulation.redshift
    central = np.array([i.central for i in sim.galaxies])
    gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies])[central]
    gal_sfr = np.array([i.sfr.in_units('Msun/Gyr') for i in sim.galaxies])[central]
    gal_ssfr = np.log10(gal_sfr / gal_sm) 

    gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies])[central]

    fractions = read_phases(fracdata_dir+'available_mass_fraction.h5', all_phases)

    frac_stats = {}
    mass_bins = get_bin_edges(min_mass, max_mass, dm)
    frac_stats['smass_bins'] = get_bin_middle(np.append(mass_bins, mass_bins[-1] + dm))  

    mask = np.array([True] * len(gal_sm))
    frac_stats['all'] = get_phase_stats(gal_sm, gal_pos, fractions, mask, all_phases, mass_bins, boxsize, logresults=False)

    mask = gal_ssfr > quench
    frac_stats['star_forming'] = get_phase_stats(gal_sm, gal_pos, fractions, mask, all_phases, mass_bins, boxsize, logresults=False)

    mask = gal_ssfr < quench
    frac_stats['quenched'] = get_phase_stats(gal_sm, gal_pos, fractions, mask, all_phases, mass_bins, boxsize, logresults=False)

    write_phase_stats(frac_stats_file, frac_stats, all_phases, stats)

fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax = ax.flatten()

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['all'][phase]['median']
    
running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue
    ax[0].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['all'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['all'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['star_forming'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue    
    ax[1].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['star_forming'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['star_forming'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['quenched'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    if phase == 'Dust':
        continue    
    ax[2].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['quenched'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i], alpha=alpha)
    running_total += frac_stats['quenched'][phase]['median'] / total

ax[0].set_title('All')
ax[1].set_title('Star forming')
ax[2].set_title('Quenched')
for i in range(3):
    ax[i].set_xlim(frac_stats['smass_bins'][0], frac_stats['smass_bins'][-1])
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
    ax[i].set_ylabel(r'$f_{\rm Total}$')
ax[2].legend(loc=2, fontsize=11)
plt.savefig(savedir+model+'_'+wind+'_'+snap+'_avail_fracs_peeples.png', bbox_inches = 'tight')
plt.clf()
