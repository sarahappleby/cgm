import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

min_mass = 9.5
max_mass = 12.
dm = 0.2 # dex

snap = '151'
wind = 's50'
model = 'm100n1024'

if model == 'm100n1024':
    boxsize = 100000.
elif model == 'm50n512':
    boxsize = 50000.

fracdata_dir = '/home/sarah/cgm/budgets/data/'
savedir = '/home/sarah/cgm/budgets/plots/'
# fracdata_dir = '/home/sapple/cgm/budgets/data/'
# savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)', 
              'ISM', 'Wind', 'Dust', 'Stars']
plot_phases_labels = [r'Cool CGM $(T < T_{\rm photo})$', r'Warm CGM $(T_{\rm photo} < T < T_{\rm vir})$', 
                      r'Hot CGM $(T > T_{\rm vir})$', 'ISM', 'Wind', 'Dust', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r']
stats = ['median', 'percentile_25_75', 'cosmic_median', 'cosmic_std']

frac_stats_file = fracdata_dir+model+'_'+wind+'_'+snap+'_avail_metal_frac_stats.h5'

if os.path.isfile(frac_stats_file):

    frac_stats = read_phase_stats(frac_stats_file, plot_phases, stats)

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

    fractions = read_phases(fracdata_dir+'available_metal_fraction.h5', all_phases)

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
    ax[0].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['all'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i])
    running_total += frac_stats['all'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['star_forming'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    ax[1].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['star_forming'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i])
    running_total += frac_stats['star_forming'][phase]['median'] / total

total = np.zeros(len(frac_stats['smass_bins']))
for phase in plot_phases:
    total += frac_stats['quenched'][phase]['median']

running_total = np.zeros(len(frac_stats['smass_bins']))
for i, phase in enumerate(plot_phases):
    ax[2].fill_between(frac_stats['smass_bins'], running_total, running_total + (frac_stats['quenched'][phase]['median'] / total), 
                        color=colours[i], label=plot_phases_labels[i])
    running_total += frac_stats['quenched'][phase]['median'] / total
    
ax[0].annotate('All', xy=(0.05, 0.9), xycoords='axes fraction',size=16,bbox=dict(boxstyle="round", fc="w"))
ax[1].annotate('SF', xy=(0.05, 0.9), xycoords='axes fraction',size=16,bbox=dict(boxstyle="round", fc="w"))
ax[2].annotate('Q', xy=(0.05, 0.9), xycoords='axes fraction',size=16,bbox=dict(boxstyle="round", fc="w"))
for i in range(3):
    ax[i].set_xlim(min_mass, frac_stats['smass_bins'][-1]+0.5*dm)
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
    ax[i].set_ylabel(r'$f_{Z\ {\rm Total}}$')
ax[1].legend(loc=4, fontsize=11)
plt.savefig(savedir+model+'_'+wind+'_'+snap+'_avail_metal_fracs_peeples.png')
plt.clf()