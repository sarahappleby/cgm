import matplotlib.pyplot as plt
import numpy as np
import h5py
import caesar
import os
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

palette_name = 'tol'
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

zdata_dir = '/home/sarah/cgm/budgets/data/'
savedir = '/home/sarah/cgm/budgets/plots/'
# zdata_dir = '/home/sapple/cgm/budgets/data/'
# savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
                'Wind', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$', 
                      r'Cool CGM $(T < T_{\rm photo})$', 'Wind', 'ISM', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:pink', 'r']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'cosmic_median', 'cosmic_std']

z_stats_file = zdata_dir+model+'_'+wind+'_'+snap+'_metallicities_stats.h5'

if os.path.isfile(z_stats_file):

    z_stats = read_phase_stats(z_stats_file, plot_phases, stats)

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

    # get the mass budget data:
    metallicities = read_phases(zdata_dir+'metallicities.h5', all_phases)

    z_stats = {}
    mass_bins = get_bin_edges(min_mass, max_mass, dm)
    z_stats['smass_bins'] = get_bin_middle(np.append(mass_bins, mass_bins[-1] + dm))   

    mask = np.array([True] * len(gal_sm))
    z_stats['all'] = get_phase_stats(gal_sm, gal_pos, metallicities, mask, all_phases, mass_bins, boxsize, logresults=True)

    mask = gal_ssfr > quench
    z_stats['star_forming'] = get_phase_stats(gal_sm, gal_pos, metallicities, mask, all_phases, mass_bins, boxsize, logresults=True)

    mask = gal_ssfr < quench
    z_stats['quenched'] = get_phase_stats(gal_sm, gal_pos, metallicities, mask, all_phases, mass_bins, boxsize, logresults=True)

    write_phase_stats(z_stats_file, z_stats, all_phases, stats)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()

for i, phase in enumerate(plot_phases):
    ax[0].errorbar(z_stats['smass_bins'], z_stats['all'][phase]['median'], yerr=z_stats['all'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[1].errorbar(z_stats['smass_bins'], z_stats['star_forming'][phase]['median'], yerr=z_stats['star_forming'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
for i, phase in enumerate(plot_phases):
    ax[2].errorbar(z_stats['smass_bins'], z_stats['quenched'][phase]['median'], yerr=z_stats['quenched'][phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])
ax[0].set_title('All')
ax[1].set_title('Star forming')
ax[2].set_title('Quenched')
for i in range(3):
    ax[i].set_xlim(min_mass, z_stats['smass_bins'][-1]+0.5*dm)
    ax[i].set_ylim(-3.5, -1.5)
    ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
    ax[i].set_ylabel(r'$Z$')
ax[0].legend(loc=4, fontsize=11, )
plt.savefig(savedir+model+'_'+wind+'_'+snap+'_metallcities.png', bbox_inches = 'tight')
plt.clf()
