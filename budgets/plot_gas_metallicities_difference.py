import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import h5py
import os
import sys
import caesar
import numpy as np 
from plotting_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
palette_name = 'tol'

solar_z = 0.0134
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex
ngals_min = 10
xoffset = 0.03
linestyles=[':', '--', '-']

snap = '151'
winds = ['s50nox', 's50nojet', 's50noagn']
model = 'm50n512'
boxsize = 50000.
wind_labels = [r'$\textrm{No-Xray - Simba}$', r'$\textrm{No-jet - Simba}$', r'$\textrm{No-AGN - Simba}$']
savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
                          'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
                          'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)','ISM']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', r'Warm CGM $(T_{\rm photo} < T < 0.5T_{\rm vir})$',
                                          r'Cool CGM $(T < T_{\rm photo})$', 'ISM']

#cmap = cm.get_cmap('plasma')
#colours = [cmap(0.25), cmap(0.5), cmap(0.75)]
#dc = 0.7 / (len(winds) - 1)
#colours = [cmap(0.15 + i*dc) for i in range(len(winds))]
colours = get_cb_colours(palette_name)[::-1]
colours = np.delete(colours, [3, 4, 5, 6])

stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

fig, ax = plt.subplots(2, 2, figsize=(13, 13))
ax = ax.flatten()

line_x = Line2D([0,1],[0,1],ls=linestyles[0], marker='o', color=colours[0])
line_jet = Line2D([0,1],[0,1],ls=linestyles[1], marker='o', color=colours[1])
line_agn = Line2D([0,1],[0,1],ls=linestyles[2], marker='o', color=colours[2])
leg_winds = ax[0].legend([line_x, line_jet, line_agn],wind_labels, loc=0)
ax[0].add_artist(leg_winds)

simba_data_dir = '/home/sapple/cgm/budgets/data/'+model+'_s50/'
simba_z_stats_file = simba_data_dir+model+'_s50_'+snap+'_metallicities_stats.h5' 
simba_z_stats = read_phase_stats(simba_z_stats_file, plot_phases, stats)
simba_mask = simba_z_stats['all']['ngals'][:] > ngals_min

for w, wind in enumerate(winds):

    data_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'/'
    z_stats_file = data_dir+model+'_'+wind+'_'+snap+'_metallicities_stats.h5'

    if os.path.isfile(z_stats_file):
        z_stats = read_phase_stats(z_stats_file, plot_phases, stats)
    else:
        print('Need to run plot_metallicities_winds first! :) ')

    if w == 0:
        z_stats['smass_bins'] -= xoffset
    if w == 2:
        z_stats['smass_bins'] += xoffset

    wind_mask = z_stats['all']['ngals'][:] > ngals_min
    mask = simba_mask * wind_mask

    for i, phase in enumerate(plot_phases):

        if w == 0:
            ax[i].axhline(0, c='k', ls=':', lw=1)
        # do the error bar point
        diff = z_stats['all'][phase]['median'] - simba_z_stats['all'][phase]['median']
        err = np.sqrt(simba_z_stats['all'][phase]['percentile_25_75']**2. + z_stats['all'][phase]['percentile_25_75']**2.)
        l1 = ax[i].errorbar(z_stats['smass_bins'][mask][0], diff[mask][0], yerr=[[err[0][mask][0]], [err[1][mask][0]]], 
                           capsize=3, color=colours[w], marker='')
        #l1[-1][0].set_linestyle(linestyles[w])
        ax[i].plot(z_stats['smass_bins'][mask], diff[mask], color=colours[w], marker='o', ls=linestyles[w])

        if w == 0:
            ax[i].set_xlim(min_mass, z_stats['smass_bins'][mask][-1]+0.5*dm)
            ax[i].set_ylim(-0.85, .7)
            ax[i].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
            ax[i].set_ylabel(r'$\Delta {\rm log} Z$')
            ax[i].set_title(plot_phases_labels[i])

plt.savefig(savedir+model+'_'+snap+'_metallcities_difference.png', bbox_inches = 'tight')
plt.clf()

