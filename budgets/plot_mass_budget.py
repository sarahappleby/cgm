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

massdata_dir = '/home/sarah/cgm/budgets/data/'
savedir = '/home/sarah/cgm/budgets/plots/'
# massdata_dir = '/home/sapple/cgm/budgets/data/'
# savedir = '/home/sapple/cgm/budgets/plots/'

all_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)',
              'Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
              'ISM', 'Wind', 'Dust', 'Stars', 'Dark matter', 'Total baryons']
plot_phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)', 
              'ISM', 'Wind', 'Dust', 'Stars']
plot_phases_labels = [r'Cool CGM $(T < T_{\rm photo})$', r'Warm CGM $(T_{\rm photo} < T < T_{\rm vir})$', 
                      r'Hot CGM $(T > T_{\rm vir})$', 'ISM', 'Wind', 'Dust', 'Stars']
colours = ['m', 'tab:orange', 'g', 'b', 'c', 'tab:pink', 'r']
stats = ['median', 'percentile_25_75', 'cosmic_median', 'cosmic_std', 'ngals']

mass_stats_file = massdata_dir+model+'_'+wind+'_'+snap+'_mass_budget_stats.h5'

if os.path.isfile(mass_stats_file):

    mass_stats = {p: {} for p in plot_phases}
    with h5py.File(mass_stats_file, 'r') as hf:
        for phase in plot_phases:
            for stat in stats:
                mass_stats[phase][stat] = hf[phase][stat][:]

        plot_bins = hf['smass_bins'][:]

else:

    mass_bins = get_bin_edges(min_mass, max_mass, dm)
    plot_bins = get_bin_middle(np.append(mass_bins, mass_bins[-1] + dm))

    # get the galaxy data:
    #caesarfile = '/home/sarah/data/caesar_snap_m12.5n128_135.hdf5'
    #sim = caesar.load(caesarfile)
    caesarfile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
    sim = caesar.quick_load(caesarfile)
    central = np.array([i.central for i in sim.galaxies])
    gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies])[central]
    gal_pos = np.array([i.pos.in_units('kpc/h') for i in sim.galaxies])[central]

    # get the mass budget data:
    mass_budget = {}
    with h5py.File(massdata_dir+'mass_budget.h5', 'r') as hf:
        for phase in all_phases:
            mass_budget[phase] = hf[phase][:]

    mass_stats = {phase: {} for phase in all_phases}
    binned_pos = bin_data(gal_sm, gal_pos, 10.**mass_bins, group_high=True)
    for phase in all_phases:
        binned_data = bin_data(gal_sm, mass_budget[phase], 10.**mass_bins, group_high=True)
        
        medians = np.zeros(len(plot_bins))
        cosmic_stds = np.zeros(len(plot_bins))
        for i in range(len(plot_bins)):
            medians[i], cosmic_stds[i] = get_cosmic_variance(binned_data[i], binned_pos[i], boxsize)

        mass_stats[phase]['cosmic_median'], mass_stats[phase]['cosmic_std'] = convert_to_log(medians, cosmic_stds)
        medians = np.array([np.nanpercentile(j, 50.) for j in binned_data])
        per25 = np.array([np.nanpercentile(j, 25.) for j in binned_data])
        per75 = np.array([np.nanpercentile(j, 75.) for j in binned_data])
        upper = per75 - medians
        lower = medians - per25
        mass_stats[phase]['median'], mass_stats[phase]['percentile_25_75'] = convert_to_log(medians, np.array([lower, upper]))
        mass_stats[phase]['ngals'] = [len(j) for j in binned_data]

    with h5py.File(mass_stats_file, 'a') as hf:
        for phase in all_phases:
            grp = hf.create_group(phase)
            for stat in stats:
                grp.create_dataset(stat, data=np.array(mass_stats[phase][stat]))

        hf.create_dataset('smass_bins', data=np.array(plot_bins))

for i, phase in enumerate(plot_phases):
    plt.errorbar(plot_bins, mass_stats[phase]['median'], yerr=mass_stats[phase]['percentile_25_75'], 
                capsize=3, color=colours[i], label=plot_phases_labels[i])

plt.legend(loc=2, fontsize=11)
plt.xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
plt.ylabel(r'$\textrm{log} (M / \textrm{M}_{\odot})$')
plt.xlim(min_mass, plot_bins[-1]+dm)
plt.savefig(savedir+model+'_'+wind+'_'+snap+'_mass_budget.png')
plt.clf()
