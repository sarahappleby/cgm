import matplotlib.pyplot as plt
import h5py
import os
import sys
import caesar
import numpy as np 
from plotting_methods import *
from get_mhalo_mstar import get_mhalo_axis_values

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=17)
palette_name = 'tol'

alpha = .8
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex
ngals_min = 10

snap = '151'
winds = ['s50', 's50nox', 's50nojet', 's50nofb']
model = 'm50n512'
boxsize = 50000.
wind_title = [r'$\textrm{Simba}$', r'$\textrm{No-Xray}$', r'$\textrm{No-jet}$', r'$\textrm{No-feedback}$']
savedir = '/disk01/sapple/cgm/budgets/plots/'
    
all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Cosmic baryon mass']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)',
                'Wind', 'Dust', 'ISM', 'Stars']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', 'Warm CGM\n' + r'$(T_{\rm photo} < T < 0.5T_{\rm vir})$', 
                      r'Cool CGM $(T < T_{\rm photo})$', 'Wind', 'Dust', 'ISM', 'Stars']
colours = ['m', 'b', 'c', 'g', 'tab:orange', 'tab:pink', 'r']
colours = get_cb_colours(palette_name)[::-1]
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

fig, ax = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
ax = ax.flatten()

for w, wind in enumerate(winds):

    if w == len(winds) -1:
        mhalo_axis, mstar_axis = get_mhalo_axis_values(min_mhalo=11., max_mhalo=15., model=model, wind=wind)
    else:
        mhalo_axis, mstar_axis = get_mhalo_axis_values(min_mhalo=11., max_mhalo=14., model=model, wind=wind)

    data_dir = '/disk01/sapple/cgm/budgets/data/'+model+'_'+wind+'_'+snap+'/'
    frac_stats_file = data_dir+model+'_'+wind+'_'+snap+'_omega_frac_stats.h5'

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

        fractions = read_phases(data_dir+'omega_mass_fraction.h5', all_phases)

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

    mask = frac_stats['all']['ngals'][:] > ngals_min

    running_total = np.zeros(len(frac_stats['smass_bins']))
    for i, phase in enumerate(plot_phases):
        if phase == 'Dust':
            continue
        ax[w].fill_between(frac_stats['smass_bins'][mask], running_total[mask], running_total[mask] + frac_stats['all'][phase]['median'][mask], 
                            color=colours[i], label=plot_phases_labels[i], alpha=alpha)
        running_total += frac_stats['all'][phase]['median']

    ax[w].set_xlim(frac_stats['smass_bins'][0], frac_stats['smass_bins'][-1])
    ax[w].set_ylim(0, 1)
    ax[w].set_xlabel(r'$\textrm{log} (M_{\star} / \textrm{M}_{\odot})$')

    new_ax = ax[w].twiny()
    new_ax.set_xlim(ax[w].get_xlim())
    new_ax.set_xticks(mstar_axis)
    new_ax.set_xticklabels(mhalo_axis.astype('int'))
    new_ax.set_xlabel(r'$\textrm{log} (M_{\rm halo} / \textrm{M}_{\odot})$')


x = [0.72, 0.64, 0.72, 0.53]
for i in range(4):

    ax[i].annotate(wind_title[i], xy=(x[i], 0.05), xycoords='axes fraction',size=17,
            bbox=dict(boxstyle='round', fc='white'))

ax[0].set_ylabel(r'$f_{\rm \Omega}$')
ax[0].legend(loc=2, fontsize=13, framealpha=0.)
fig.subplots_adjust(wspace=0.)
plt.savefig(savedir+model+'_'+snap+'_omega_fracs_peeples_winds_axis.png', bbox_inches = 'tight')
plt.clf()
