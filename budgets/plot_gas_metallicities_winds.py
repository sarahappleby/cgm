import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import os
import sys
import caesar
import numpy as np
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
palette_name = 'tol'

solar_z = 0.0134
alpha = 1.
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex
ngals_min = 10
linestyles = ['-', '--', ':', '-.']
xoffset = 0.025

snap = '151'
winds = ['s50', 's50nox', 's50nojet', 's50noagn']
model = 'm50n512'
boxsize = 50000.
wind_labels = [r'$\textbf{Simba}$', r'$\textbf{No-Xray}$', r'$\textbf{No-jet}$', r'$\textbf{No-AGN}$']

system = sys.argv[1]
if system == 'laptop':
        savedir = '/home/sarah/cgm/budgets/plots/'
elif system == 'ursa':
        savedir = '/home/sapple/cgm/budgets/plots/'


all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
                          'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
                          'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)','ISM']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$',
                                          r'Cool CGM $(T < T_{\rm photo})$', 'ISM']

colours = get_cb_colours(palette_name)[::-1]
colours = np.delete(colours, [3, 4, 6])
stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

fig, ax = plt.subplots(2, 2, figsize=(13, 13))
ax = ax.flatten()

line_sim = Line2D([0,1],[0,1],ls=linestyles[0], color='grey')
line_x = Line2D([0,1],[0,1],ls=linestyles[1], color='grey')
line_jet = Line2D([0,1],[0,1],ls=linestyles[2], color='grey')
line_agn = Line2D([0,1],[0,1],ls=linestyles[3], color='grey')

leg_winds = ax[0].legend([line_sim, line_x, line_jet, line_agn],wind_labels, loc=2, fontsize=12)
ax[0].add_artist(leg_winds)

for w, wind in enumerate(winds):

        if system == 'laptop':
                data_dir = '/home/sarah/cgm/budgets/data/'+model+'_'+wind+'/'
        elif system == 'ursa':
                data_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'/'
        z_stats_file = data_dir+model+'_'+wind+'_'+snap+'_metallicities_stats.h5'

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
                metallicities = read_phases(data_dir+'metallicities.h5', all_phases)
                for phase in all_phases:
                        metallicities[phase] /= solar_z

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

        if w == 0:
            z_stats['smass_bins'] -= 1.5* xoffset
        elif w == 1:
            z_stats['smass_bins'] -= 0.5* xoffset
        elif w == 2:
            z_stats['smass_bins'] += 0.5* xoffset
        elif w == 3:
            z_stats['smass_bins'] += 1.5* xoffset


        mask = z_stats['all']['ngals'][:] > ngals_min

        for i, phase in enumerate(plot_phases):
            l1 = ax[i].errorbar(z_stats['smass_bins'][mask], z_stats['all'][phase]['median'][mask],
                                yerr=[z_stats['all'][phase]['percentile_25_75'][0][mask], z_stats['all'][phase]['percentile_25_75'][1][mask]],
                                capsize=3, color=colours[i], ls=linestyles[w])
            l1[-1][0].set_linestyle(linestyles[w])

            if w == 0:

                ax[i].set_xlim(min_mass, z_stats['smass_bins'][-1]+0.5*dm)
                ax[i].set_ylim(-1.7, 0.3)
                ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
                ax[i].set_ylabel(r'$\textrm{log} (Z / Z_{\odot})$')
                ax[i].set_title(plot_phases_labels[i])

plt.savefig(savedir+model+'_'+snap+'_gas_metallcities_winds.png', bbox_inches = 'tight')
plt.clf()

