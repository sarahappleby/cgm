
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

if __name__ == '__main__':

    # set some parameters
    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    plot_line_x = [0.8, 0.8, 0.72, 0.73, 0.74, 0.74]
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

    model = sys.argv[1]
    wind = sys.argv[2]

    plot_dir = 'plots/'
    r200_scaled = True
    background = 'uvb_fg20'

    sim_colors, cos_colors = get_tol_colors()

    plot_name = model+'_'+wind +'_'+background+'_rho_cfrac'
    if r200_scaled:
        scale_str = '_scaled'
        plot_name += scale_str
        xlabel = r'$\rho / r_{200}$'
    else:
        scale_str = ''
        xlabel = r'$\rho (\textrm{kpc})$'
    if plot_name[-1] == '_': plot_name = plot_name[:-1]
    plot_name += '.png'

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/'
    cos_halos_file = basic_dir+'data/cos_halos_obs_cfrac_data'+scale_str+'.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = basic_dir+'data/cos_dwarfs_obs_cfrac_data'+scale_str+'.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)
    sim_halos_file = basic_dir+'data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_cfrac_data'+scale_str+'.h5'
    sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
    sim_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_cfrac_data'+scale_str+'.h5'
    sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey='row', sharex='col')
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        # choose the survey and some params
        if survey == 'dwarfs':
            cos_plot_dict = cos_dwarfs_plot_dict.copy()
            sim_plot_dict = sim_dwarfs_plot_dict.copy()
            label = 'COS-Dwarfs'
            cos_marker='^'
        elif survey == 'halos':
            cos_plot_dict = cos_halos_plot_dict.copy()
            sim_plot_dict = sim_halos_plot_dict.copy()
            label = 'COS-Halos'
            cos_marker = 'o'

        #l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['cfrac_'+lines[i]+'_sf'],
        #                    yerr=sim_plot_dict['cfrac_'+lines[i]+'_poisson_sf'], capsize=4, c=sim_colors[0],
        #                    marker='o', ls='--')
        #l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['cfrac_'+lines[i]+'_q'],
        #                    yerr=sim_plot_dict['cfrac_'+lines[i]+'_poisson_q'], capsize=4, c=sim_colors[1],
        #                    marker='o', ls='--')
        l1, = ax[i].plot(sim_plot_dict['plot_bins_sf'], sim_plot_dict['cfrac_'+lines[i]+'_sf'], c=sim_colors[0], ls='-', marker='', lw=2)
        ax[i].fill_between(sim_plot_dict['plot_bins_sf'],
                           sim_plot_dict['cfrac_'+lines[i]+'_sf'] - sim_plot_dict['cfrac_'+lines[i]+'_poisson_sf'],
                           sim_plot_dict['cfrac_'+lines[i]+'_sf'] + sim_plot_dict['cfrac_'+lines[i]+'_poisson_sf'],
                           color=sim_colors[0], alpha=0.25)
        l2, = ax[i].plot(sim_plot_dict['plot_bins_q'], sim_plot_dict['cfrac_'+lines[i]+'_q'], c=sim_colors[1], ls='-', marker='', lw=2)
        ax[i].fill_between(sim_plot_dict['plot_bins_q'],
                           sim_plot_dict['cfrac_'+lines[i]+'_q'] - sim_plot_dict['cfrac_'+lines[i]+'_poisson_q'],
                           sim_plot_dict['cfrac_'+lines[i]+'_q'] + sim_plot_dict['cfrac_'+lines[i]+'_poisson_q'],
                           color=sim_colors[1], alpha=0.25)

        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=14, framealpha=0.)


        c1 = ax[i].errorbar(cos_plot_dict['plot_bins_sf'], cos_plot_dict['cfrac_'+lines[i]+'_sf'], 
                            yerr=cos_plot_dict['cfrac_'+lines[i]+'_poisson_sf'], xerr=cos_plot_dict['xerr_sf'], 
                            capsize=4, c=cos_colors[0], mec=cos_colors[0], mfc='white', marker=cos_marker, markersize=8, ls='', label=label+' SF')
        c2 = ax[i].errorbar(cos_plot_dict['plot_bins_q'], cos_plot_dict['cfrac_'+lines[i]+'_q'], 
                            yerr=cos_plot_dict['cfrac_'+lines[i]+'_poisson_q'], xerr=cos_plot_dict['xerr_q'], 
                            capsize=4, c=cos_colors[1], mec=cos_colors[1], mfc='white', marker=cos_marker, markersize=8, ls='', label=label+' Q')
        for c in range(2):
            c1[-1][c].set_alpha(alpha=0.5)
            c2[-1][c].set_alpha(alpha=0.5)
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=14, loc=1, framealpha=0.)
        
        ax[i].annotate(plot_lines[i], xy=(plot_line_x[i], 0.73), xycoords='axes fraction',
                        bbox=dict(boxstyle='round', fc='white'))
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylim(-0.1, 1.2)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

        if i==0:
            ax[i].add_artist(leg2)
        if i in [0, 3]:
            ax[i].set_ylabel(r'$f_{\textrm{cov}}$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')

