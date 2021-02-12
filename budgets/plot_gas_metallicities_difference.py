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
plt.rc('font', family='serif', size=17)
palette_name = 'tol'

solar_z = 0.0134
min_mass = 9.
max_mass = 12.
dm = 0.25 # dex
ngals_min = 10
xoffset = 0.03
linestyles = ['--', '-', ':', '-.']

snap = '151'
winds = ['s50nox', 's50nojet', 's50noagn', 's50nofb']
model = 'm50n512'
wind_labels = [r'$\textrm{No-Xray - Simba}$', r'$\textrm{No-jet - Simba}$', r'$\textrm{No-AGN - Simba}$', r'$\textrm{No-feedback - Simba}$']
savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Hot CGM (T > 0.5Tvir)',
                          'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
                          'ISM', 'Wind', 'Dust', 'Stars', 'Total baryons']
plot_phases = ['Hot CGM (T > 0.5Tvir)', 'Warm CGM (Tphoto < T < 0.5Tvir)', 'Cool CGM (T < Tphoto)','ISM']
plot_phases_labels = [r'Hot CGM $(T > 0.5T_{\rm vir})$', 'Warm CGM\n'+r'$(T_{\rm photo} < T < 0.5T_{\rm vir})$',
                                          r'Cool CGM $(T < T_{\rm photo})$', 'ISM']

#cmap = cm.get_cmap('plasma')
#colours = [cmap(0.25), cmap(0.5), cmap(0.75)]
#dc = 0.8 / (len(winds) - 1)
#colours = [cmap(0.1 + i*dc) for i in range(len(winds))][::-1]
colours = get_cb_colours(palette_name)[::-1]
colours = np.delete(colours, [3, 4, 5, ])
colours = np.roll(colours, 1)[::-1]

stats = ['median', 'percentile_25_75', 'std', 'cosmic_median', 'cosmic_std']

fig, ax = plt.subplots(1, 4, figsize=(15, 5.5), sharey='row')
ax = ax.flatten()

wind_lines = []
for w in range(len(winds)):
    wind_lines.append(Line2D([0,1],[0,1],ls=linestyles[w], color=colours[w]))
leg_wind = ax[3].legend(wind_lines,wind_labels, loc=4, fontsize=16, framealpha=0.)
ax[3].add_artist(leg_wind)

simba_data_dir = '/home/sapple/cgm/budgets/data/'+model+'_s50_151/'
simba_z_stats_file = simba_data_dir+model+'_s50_'+snap+'_metallicities_stats.h5' 
simba_z_stats = read_phase_stats(simba_z_stats_file, plot_phases, stats)
simba_mask = simba_z_stats['all']['ngals'][:] > ngals_min

for w, wind in enumerate(winds):

    data_dir = '/home/sapple/cgm/budgets/data/'+model+'_'+wind+'_'+snap+'/'
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
        if (wind == 's50nofb') and (i < 2) :
            diff[diff < -2.] = -2.
            l1 = ax[i].errorbar(z_stats['smass_bins'][mask][0], diff[mask][0], 
                               yerr=[[err[0][mask][4]], [err[1][mask][4]]], 
                               capsize=3, color=colours[w], marker='')
        else:
            l1 = ax[i].errorbar(z_stats['smass_bins'][mask][0], diff[mask][0], 
                               yerr=[[err[0][mask][0]], [err[1][mask][0]]], 
                               capsize=3, color=colours[w], marker='') 
        ax[i].plot(z_stats['smass_bins'][mask], diff[mask], color=colours[w], ls=linestyles[w])

        if w == 0:
            ax[i].set_xlim(min_mass, z_stats['smass_bins'][mask][-1]+0.5*dm)
            ax[i].set_ylim(-2.1, .7)
            ax[i].set_xlabel(r'$\textrm{log} (M_{\star} / \textrm{M}_{\odot})$')

x = [0.21, 0.28, 0.21, 0.81]
ax[0].annotate(plot_phases_labels[0], xy=(x[0], 0.92), xycoords='axes fraction',size=15,
        bbox=dict(boxstyle='round', fc='white'))
ax[1].annotate(plot_phases_labels[1], xy=(x[1], 0.87), xycoords='axes fraction',size=15,
        bbox=dict(boxstyle='round', fc='white'))
ax[2].annotate(plot_phases_labels[2], xy=(x[2], 0.92), xycoords='axes fraction',size=15,
        bbox=dict(boxstyle='round', fc='white'))
ax[3].annotate(plot_phases_labels[3], xy=(x[3], 0.92), xycoords='axes fraction',size=15,
        bbox=dict(boxstyle='round', fc='white'))

ax[0].set_ylabel(r'$\Delta {\rm log} Z$')
fig.subplots_adjust(wspace=0.)
plt.savefig(savedir+model+'_'+snap+'_metallcities_difference.png', bbox_inches = 'tight',
            metadata={'creator': 'plot_gas_metallicities_difference.py'})
plt.clf()

