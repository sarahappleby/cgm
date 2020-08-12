
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from physics import compute_path_length
from analysis_methods import *

sys.path.append('../cos_samples/')
from get_cos_info import make_cos_dict, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':
    
    # set some parameters
    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    wave_rest = [1215., 1215., 2796., 1206., 1548., 1031.]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    model = 'm100n1024'
    wind = 's50'
    
    plot_dir = 'plots/'
    velocity_width = 300. # km/s
    r200_scaled = True
    do_equal_bins = False
    norients = 8
    ngals_each = 5
    mlim = np.log10(5.8e8) # lower limit of M*

    nbins_halos_q = 2
    nbins_halos_sf = 3
    nbins_dwarfs_q = 1
    nbins_dwarfs_sf = 3

    # get the quenching thresholds for each survey
    halos_z = 0.25; halos_quench = quench_thresh(halos_z)
    dwarfs_z = 0.; dwarfs_quench = quench_thresh(dwarfs_z)

    plot_name = model+'_'+wind +'_rho_path_abs'
    if r200_scaled:
        plot_name += '_scaled'
    if do_equal_bins:
        plot_name += '_equal_bins'
    plot_name += '.png'

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    # read in COS sample data, masked for low mass galaxies
    cos_halos_dict, cos_halos_mmask = make_cos_dict('halos', mlim, r200_scaled)
    cos_dwarfs_dict, cos_dwarfs_mmask = make_cos_dict('dwarfs', mlim, r200_scaled)

    # create the dicts to hold the simulation sample data
    sim_halos_dict = read_simulation_sample(model, wind, '137', 'halos', norients, lines, r200_scaled)
    sim_halos_dict['rho'] = np.repeat(cos_halos_dict['rho'], norients*ngals_each)

    sim_dwarfs_dict = read_simulation_sample(model, wind, '151', 'dwarfs', norients, lines, r200_scaled)
    sim_dwarfs_dict['rho'] = np.repeat(cos_dwarfs_dict['rho'], norients*ngals_each)


    # rescaled the x axis by r200
    if r200_scaled:
        cos_halos_dict['dist'] = cos_halos_dict['rho'] / cos_halos_dict['r200']
        cos_dwarfs_dict['dist'] = cos_dwarfs_dict['rho'] / cos_dwarfs_dict['r200']
        sim_halos_dict['dist'] = sim_halos_dict['rho'] / sim_halos_dict['r200']
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'] / sim_dwarfs_dict['r200']
        xlabel = r'$\rho / r_{200}$'
    else:
        cos_halos_dict['dist'] = cos_halos_dict['rho'].copy()
        cos_dwarfs_dict['dist'] = cos_dwarfs_dict['rho'].copy()
        sim_halos_dict['dist'] = sim_halos_dict['rho'].copy()
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'].copy()
        xlabel = r'$\rho (\textrm{kpc})$'

    # get the bins for the COS data - these nbins ensure there are roughly ~8 galaxies in each bin
    mask = (cos_halos_dict['ssfr'] > halos_quench)
    cos_halos_dict['dist_bins_sf'], cos_halos_dict['plot_bins_sf'] = do_bins(cos_halos_dict['dist'][mask], nbins_halos_sf)
    cos_halos_dict['dist_bins_q'], cos_halos_dict['plot_bins_q'] = do_bins(cos_halos_dict['dist'][~mask], nbins_halos_q)

    mask = (cos_dwarfs_dict['ssfr'] > dwarfs_quench)
    cos_dwarfs_dict['dist_bins_sf'], cos_dwarfs_dict['plot_bins_sf'] = do_bins(cos_dwarfs_dict['dist'][mask], nbins_dwarfs_sf)
    cos_dwarfs_dict['dist_bins_q'], cos_dwarfs_dict['plot_bins_q'] = do_bins(cos_dwarfs_dict['dist'][~mask], nbins_dwarfs_q)

    sim_halos_dict = get_equal_bins(sim_halos_dict, 'halos', r200_scaled)
    sim_dwarfs_dict = get_equal_bins(sim_dwarfs_dict, 'dwarfs', r200_scaled)

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        # choose the survey and some params
        if survey == 'dwarfs':
            cos_dict = cos_dwarfs_dict.copy()
            sim_dict = sim_dwarfs_dict.copy()
            mass_mask = cos_dwarfs_mmask.copy()
            label = 'COS-Dwarfs'
            quench = dwarfs_quench + 0.
            z = dwarfs_z + 0.
        elif survey == 'halos':
            cos_dict = cos_halos_dict.copy()
            sim_dict = sim_halos_dict.copy()
            mass_mask = cos_halos_mmask.copy()
            label = 'COS-Halos'
            quench = halos_quench + 0.
            z = halos_z + 0.

        # Find the path length
        sim_dict['path_length_'+lines[i]] = compute_path_length(sim_dict['vgal_position'], velocity_width, wave_rest[i], z)

        # removing COS-Dwarfs galaxy 3 for the Lya stuff
        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            mass_mask = np.delete(mass_mask, 3)
            for k in cos_dict.keys():
                if k in ['dist_bins_sf', 'plot_bins_sf', 'dist_bins_q',  'plot_bins_q']: continue
                else: cos_dict[k] = np.delete(cos_dict[k], 3)
            for k in sim_dict.keys():
                if k in ['dist_bins_sf', 'plot_bins_sf', 'dist_bins_q',  'plot_bins_q']: continue
                else: sim_dict[k] = np.delete(sim_dict[k], np.arange(3*norients*ngals_each, 4*norients*ngals_each), axis=0)

        # get binned medians for the simulation sample
        sim_sf_path_abs, sim_sf_err = sim_binned_path_abs(sim_dict, (sim_dict['ssfr'] > quench), sim_dict['dist_bins_sf'], det_thresh[i], lines[i], boxsize)
        sim_q_path_abs, sim_q_err = sim_binned_path_abs(sim_dict, (sim_dict['ssfr'] < quench), sim_dict['dist_bins_q'], det_thresh[i], lines[i], boxsize)

        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            cos_dict['EW'], cos_dict['EWerr'], cos_dict['EW_less_than'] = get_cos_dwarfs_civ() #in mA
            cos_dict['EW'] /= 1000.
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            cos_dict['EW'], cos_dict['EWerr'] = get_cos_dwarfs_lya() # in mA
            cos_dict['EW'] /= 1000.
            cos_dict['EW'] = np.delete(cos_dict['EW'], 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
        elif (survey == 'halos'):
            cos_dict['EW'], cos_dict['EWerr'] = read_halos_data(lines[i])
            cos_dict['EW'] = np.abs(cos_dict['EW'])
        cos_dict['EW'] = cos_dict['EW'][mass_mask]

        if survey == 'halos':
            ew_mask = cos_dict['EW'] > 0.
            for k in ['rho', 'mass', 'r200', 'ssfr', 'dist', 'EW', 'EWerr']:
                cos_dict[k] = cos_dict[k][ew_mask]
        
        cos_dict['path_length'] = np.repeat(sim_dict['path_length_'+lines[i]][0], len(cos_dict['EW']))
        
        cos_sf_path_abs = cos_binned_path_abs(cos_dict, (cos_dict['ssfr'] > quench), cos_dict['dist_bins_sf'], det_thresh[i])
        cos_q_path_abs = cos_binned_path_abs(cos_dict, (cos_dict['ssfr'] < quench), cos_dict['dist_bins_q'], det_thresh[i])
       
        cos_sf_path_abs[cos_sf_path_abs == 0.] = 10**1.6
        cos_q_path_abs[cos_q_path_abs == 0.] = 10**1.6

        cos_sf_xerr = get_xerr_from_bins(cos_dict['dist_bins_sf'], cos_dict['plot_bins_sf'])
        cos_q_xerr = get_xerr_from_bins(cos_dict['dist_bins_q'], cos_dict['plot_bins_q'])

        c1 = ax[i].errorbar(cos_dict['plot_bins_sf'], np.log10(cos_sf_path_abs), xerr=cos_sf_xerr, capsize=4, c='c', marker='', ls='')
        c2 = ax[i].errorbar(cos_dict['plot_bins_q'], np.log10(cos_q_path_abs), xerr=cos_q_xerr, capsize=4, c='m', marker='', ls='')
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        l1 = ax[i].errorbar(sim_dict['plot_bins_sf'], sim_sf_path_abs, yerr=sim_sf_err, c='b', marker='o', ls='--')
        l2 = ax[i].errorbar(sim_dict['plot_bins_q'], sim_q_path_abs, yerr=sim_q_err, c='r', marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z),\ $' + plot_lines[i])
       
        ax[i].set_ylim(1.5, 2.9)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

        if i==0:
            ax[i].add_artist(leg1)

    plt.savefig(plot_dir+plot_name)



