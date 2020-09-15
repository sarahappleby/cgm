
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np

from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':
    
    # set some parameters
    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    model = sys.argv[1]
    wind = sys.argv[2]
    
    plot_dir = 'plots/'
    r200_scaled = True
    background = 'uvb_fg20'

    sim_colors, cos_colors = get_tol_colors()

    plot_name = model+'_'+wind +'_'+background+'_rho_path_abs'
    if r200_scaled:
        plot_name += '_scaled'
    plot_name += '.png'

    if r200_scaled:
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    
    cos_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_obs_path_abs_data.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_obs_path_abs_data.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)
    sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_data.h5'
    sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
    sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_data.h5'
    sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        # choose the survey and some params
        if survey == 'dwarfs':
            cos_plot_dict = cos_dwarfs_plot_dict.copy()
            sim_plot_dict = sim_dwarfs_plot_dict.copy()
            label = 'COS-Dwarfs'
        elif survey == 'halos':
            cos_plot_dict = cos_halos_plot_dict.copy()
            sim_plot_dict = sim_halos_plot_dict.copy()
            label = 'COS-Halos'
        
        c1 = ax[i].errorbar(cos_plot_dict['plot_bins_sf'], cos_plot_dict['path_abs_'+lines[i]+'_sf'], 
                            yerr=cos_plot_dict['path_abs_'+lines[i]+'_std_sf'], xerr=cos_plot_dict['xerr_sf'], 
                            capsize=4, c=cos_colors[0], marker='s', markersize=4, ls='')
        c2 = ax[i].errorbar(cos_plot_dict['plot_bins_q'], cos_plot_dict['path_abs_'+lines[i]+'_q'], 
                            yerr=cos_plot_dict['path_abs_'+lines[i]+'_std_q'], xerr=cos_plot_dict['xerr_q'],
                            capsize=4, c=cos_colors[1], marker='s', markersize=4, ls='')
        for c in range(2):
            c1[-1][c].set_alpha(alpha=0.5)
            c2[-1][c].set_alpha(alpha=0.5)
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'], 
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf'], 
                            c=sim_colors[0], capsize=4, marker='o', ls='--')
        l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['path_abs_'+lines[i]+'_q'], 
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q'],
                            c=sim_colors[1], capsize=4, marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z),\ $' + plot_lines[i])       
        ax[i].set_ylim(0.7, 3.)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

        if i==0:
            ax[i].add_artist(leg1)

    plt.savefig(plot_dir+plot_name)



