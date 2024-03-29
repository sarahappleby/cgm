
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

if __name__ == '__main__':


    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    wave_rest = [1215., 1215., 2796., 1206., 1548., 1031.]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

    model = 'm50n512'
    winds = ['s50nox', 's50nojet']
    wind_labels = [r'$\textrm{No-Xray - Simba}$', r'$\textrm{No-jet - Simba}$']
    ls = ['--', '-']
    markers = ['v', 'o']
    ylim = 0.5
    xoffset = 0.035
    r200_scaled = True
    background = 'uvb_fg20'
    lower_lim = -1.

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+background+'_winds_rho_path_abs_thresh_difference'
    if r200_scaled:
        scale_str = '_scaled'
        plot_name += scale_str
        xlabel = r'$\rho / r_{200}$'
    else:
        scale_str = ''
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(2, 3, figsize=(21, 12.5))
    ax = ax.flatten()

    line_x = Line2D([0,1],[0,1],ls=ls[0], marker=markers[0], color='grey', mec='grey', mfc='white', markersize=8)
    line_jet = Line2D([0,1],[0,1],ls=ls[1], marker=markers[1], color='grey', mec='grey', mfc='white', markersize=8)

    leg_winds = ax[0].legend([line_x, line_jet],wind_labels, loc=2, fontsize=16, framealpha=0.)
    ax[0].add_artist(leg_winds)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=4, fontsize=16, framealpha=0.)
    ax[0].add_artist(leg_color)

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/'
    simba_halos_file = basic_dir+'data/cos_halos_'+model+'_s50j7k_137_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
    simba_halos_plot_dict = read_dict_from_h5(simba_halos_file)
    simba_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_s50j7k_151_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
    simba_dwarfs_plot_dict = read_dict_from_h5(simba_dwarfs_file)

    for j, wind in enumerate(winds):

        sim_halos_file = basic_dir+'data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        if j == 0:
            sim_halos_plot_dict['plot_bins_sf'] -= xoffset
            sim_halos_plot_dict['plot_bins_q'] -= xoffset
            sim_dwarfs_plot_dict['plot_bins_sf'] -= xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] -= xoffset
        elif j == 2:
            sim_halos_plot_dict['plot_bins_sf'] += xoffset
            sim_halos_plot_dict['plot_bins_q'] += xoffset
            sim_dwarfs_plot_dict['plot_bins_sf'] += xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] += xoffset

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                simba_plot_dict = simba_dwarfs_plot_dict
                label = 'COS-Dwarfs'
                x = 0.72
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                simba_plot_dict = simba_halos_plot_dict
                label = 'COS-Halos'
                x = 0.75
           
            if j == 0:
                ax[i].axhline(0, c='k', ls=':', lw=1)

            diff = sim_plot_dict['path_abs_'+lines[i]+'_sf'] - simba_plot_dict['path_abs_'+lines[i]+'_sf']
            err = np.sqrt(sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf']**2 + simba_plot_dict['path_abs_'+lines[i]+'_cv_std_sf']**2)
            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], diff,
                            yerr=err,
                            c=sim_colors[0], mec=sim_colors[0], mfc='white', markersize=8, marker=markers[j], 
                            ls=ls[j], capsize=4)
            l1[-1][0].set_linestyle(ls[j])

            empty_mask = ~np.isnan(sim_plot_dict['path_abs_'+lines[i]+'_q'])
            lower_lim_array = np.array([lower_lim] * len(empty_mask))
            diff = sim_plot_dict['path_abs_'+lines[i]+'_q'] - simba_plot_dict['path_abs_'+lines[i]+'_q']
            err = np.sqrt(sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q']**2 + simba_plot_dict['path_abs_'+lines[i]+'_cv_std_q']**2) 
            diff[~empty_mask] = lower_lim
            err[~empty_mask] = np.nan

            ax[i].plot(sim_plot_dict['plot_bins_q'], diff,
                            c=sim_colors[1], marker='', ls=ls[j])
            ax[i].plot(sim_plot_dict['plot_bins_q'][~empty_mask], diff[~empty_mask],
                            c=sim_colors[1], markersize=15, marker='$\downarrow$', ls='')
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'][empty_mask], diff[empty_mask],
                            yerr=err[empty_mask],
                            c=sim_colors[1], mec=sim_colors[1], mfc='white', markersize=8, marker=markers[j], 
                            ls='', capsize=4)
            l2[-1][0].set_linestyle(ls[j])

            ax[i].annotate(label, xy=(x, 0.91), xycoords='axes fraction',size=16, 
                            bbox=dict(boxstyle='round', fc='none', edgecolor='none')) 
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(r'$\Delta {\rm log}\ ({\rm dEW}/ {\rm d} z),\ $'+plot_lines[i], labelpad=0)
            ax[i].set_ylim(-1.1, 2.)
            if r200_scaled:
                ax[i].set_xlim(0, 1.5)
            else:
                ax[i].set_xlim(25, 120)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight') 
