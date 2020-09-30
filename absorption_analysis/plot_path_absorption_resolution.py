
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = ['dwarfs'] * 6
    lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770' ]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', r'$\textrm{SiIII}1206$', 
                  r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    models = ['m100n1024', 'm25n512']
    wind = 's50'
    background = 'uvb_fg20'
 
    res_labels = [r'$\textrm{Simba}$', r'$\textrm{High-res}$']
    linestyles = ['--', ':']
    markers = ['D', 'v']
    ylim = 0.5
    xoffset = 0.025
    r200_scaled = True

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = 'resolution_rho_path_abs'
    #plot_name += '_'+cos_survey[0] +'_only'
    if r200_scaled:
        plot_name += '_scaled'
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(2, 3, figsize=(17.5, 12.5))
    ax = ax.flatten()

    line_m100 = Line2D([0,1],[0,1],ls=linestyles[0], marker=markers[0], color='grey')
    line_m25 = Line2D([0,1],[0,1],ls=linestyles[1], marker=markers[1], color='grey')

    leg_res = ax[0].legend([line_m100, line_m25], res_labels, loc=4, fontsize=12)
    ax[0].add_artist(leg_res)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=12)
    ax[0].add_artist(leg_color)

    cos_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_obs_path_abs_data.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)

    for m, model in enumerate(models):

        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_data.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        if m == 0:
            sim_dwarfs_plot_dict['plot_bins_sf'] -= xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] -= xoffset
        elif m == 1:
            sim_dwarfs_plot_dict['plot_bins_sf'] += xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] += xoffset

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                cos_plot_dict = cos_dwarfs_plot_dict
                label = 'COS-Dwarfs'
                x = 0.75
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                cos_plot_dict = cos_halos_plot_dict
                label = 'COS-Halos'
                x = 0.77

            """
            if (b == 2) & ('path_abs_'+lines[i]+'_sf' in list(cos_plot_dict.keys())):
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
            """

            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'],
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf'],
                            c=sim_colors[0], markersize=6, marker=markers[m], ls=linestyles[m], capsize=4)
            l1[-1][0].set_linestyle(linestyles[m])
            empty_mask = ~np.isnan(sim_plot_dict['path_abs_'+lines[i]+'_q'])
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'][empty_mask], sim_plot_dict['path_abs_'+lines[i]+'_q'][empty_mask],
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q'][empty_mask],
                            c=sim_colors[1], markersize=6, marker=markers[m], ls=linestyles[m], capsize=4)
            l2[-1][0].set_linestyle(linestyles[m])
            #ax[i].annotate(label, xy=(x, 0.91), xycoords='axes fraction',size=12,
            #                    bbox=dict(boxstyle='round', fc='white', edgecolor='lightgrey'))
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z)\ $' + plot_lines[i])
            ax[i].set_ylim(0.7, 3.0)
            if r200_scaled:
                ax[i].set_xlim(0, 1.5)
            else:
                ax[i].set_xlim(25, 120)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')
