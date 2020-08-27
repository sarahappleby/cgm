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
    det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

    model = sys.argv[1]
    wind = sys.argv[2]
    
    plot_dir = 'plots/'
    r200_scaled = True
    ylim = 0.7
    background = 'uvb_hm12'

    sim_colors, cos_colors = get_tol_colors()

    # set plot name according to parameters
    plot_name = model+'_'+wind +'_'+background+'_rho_ew_med'
    if r200_scaled:
        plot_name += '_scaled'
    if plot_name[-1] == '_': plot_name = plot_name[:-1]
    plot_name += '.png'

    # rescaled the x axis by r200
    if r200_scaled:        
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'

    cos_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_obs_ew_med_data.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_obs_ew_med_data.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)
    sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_ew_med_data.h5'
    sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
    sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_ew_med_data.h5'
    sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        # choose the survey and some params
        if survey == 'dwarfs':
            cos_plot_dict = cos_dwarfs_plot_dict
            sim_plot_dict = sim_dwarfs_plot_dict
            label = 'COS-Dwarfs'
        elif survey == 'halos':
            cos_plot_dict = cos_halos_plot_dict
            sim_plot_dict = sim_halos_plot_dict
            label = 'COS-Halos'
        
        c1 = ax[i].errorbar(cos_plot_dict['plot_bins_sf'], cos_plot_dict['EW_'+lines[i]+'_med_sf'], xerr=cos_plot_dict['xerr_sf'], 
                            yerr=[cos_plot_dict['EW_'+lines[i]+'_per25_sf'], cos_plot_dict['EW_'+lines[i]+'_per75_sf']], 
                            capsize=4, c=cos_colors[0], marker='s', markersize=4, ls='', label=label+' SF')
        c2 = ax[i].errorbar(cos_plot_dict['plot_bins_q'], cos_plot_dict['EW_'+lines[i]+'_med_q'], xerr=cos_plot_dict['xerr_q'], 
                            yerr=[cos_plot_dict['EW_'+lines[i]+'_per25_q'], cos_plot_dict['EW_'+lines[i]+'_per75_q']], 
                            capsize=4, c=cos_colors[1], marker='s', markersize=4, ls='', label=label+' Q')
        for c in range(2):
            c1[-1][c].set_alpha(alpha=0.5)
            c2[-1][c].set_alpha(alpha=0.5)
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        # plot the Simba data as lines
        l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['EW_'+lines[i]+'_med_sf'], 
                            yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_sf'], 
                            capsize=4, c=sim_colors[0], marker='o', ls='--')
        l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['EW_'+lines[i]+'_med_q'], 
                            yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_q'], 
                            capsize=4, c=sim_colors[1], marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)
            
        ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
        ax[i].set_ylim(-2, ylim)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

        if i==0:
            ax[i].add_artist(leg1)

    plt.savefig(plot_dir+plot_name)
