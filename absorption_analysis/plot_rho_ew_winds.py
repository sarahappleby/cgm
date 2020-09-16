import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$',r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

    model = 'm50n512'
    winds = ['s50j7k', 's50nox', 's50nojet']
    wind_labels = [r'$\textrm{Simba}$', r'$\textrm{No-Xray}$', r'$\textrm{No-jet}$']
    ls = ['-', '--', ':']
    markers = ['o', 'D', 's']
    ylim = 0.5
    xoffset = 0.035
    r200_scaled = True
    background = 'uvb_fg20'
    lower_lim = -2.

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+background+'_winds_rho_ew'
    if r200_scaled:
        plot_name += '_scaled'
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(2, 3, figsize=(17.5, 12.5))
    ax = ax.flatten()

    line_sim = Line2D([0,1],[0,1],ls=ls[0], marker=markers[0], color='grey')
    line_x = Line2D([0,1],[0,1],ls=ls[1], marker=markers[1], color='grey')
    line_jet = Line2D([0,1],[0,1],ls=ls[2], marker=markers[2], color='grey')

    leg_winds = ax[0].legend([line_sim, line_x, line_jet],wind_labels, loc=4, fontsize=12)
    ax[0].add_artist(leg_winds)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=12)
    ax[0].add_artist(leg_color)

    for j, wind in enumerate(winds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_ew_med_data.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_ew_med_data.h5'
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
                label = 'COS-Dwarfs'
                x = 0.75
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                label = 'COS-Halos'
                x = 0.78

            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['EW_'+lines[i]+'_med_sf'], 
                                yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_sf'], 
                                capsize=4, c=sim_colors[0], markersize=6, marker=markers[j], linestyle=ls[j], label='Simba SF')
            l1[-1][0].set_linestyle(ls[j])
            
            empty_mask = ~np.isnan(sim_plot_dict['EW_'+lines[i]+'_med_q'])
            lower_lim_array = np.array([lower_lim] * len(empty_mask))

            ax[i].plot(sim_plot_dict['plot_bins_q'][empty_mask], sim_plot_dict['EW_'+lines[i]+'_med_q'][empty_mask],
                            c=sim_colors[1], markersize=6, marker=markers[j], ls='')
            ax[i].plot(sim_plot_dict['plot_bins_q'][~empty_mask], lower_lim_array[~empty_mask],
                            c=sim_colors[1], markersize=15, marker='$\downarrow$', ls='')
            sim_plot_dict['EW_'+lines[i]+'_med_q'][~empty_mask] = lower_lim
            sim_plot_dict['EW_'+lines[i]+'_cosmic_std_q'][~empty_mask] = np.nan
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['EW_'+lines[i]+'_med_q'],
                            yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_q'], capsize=4, c=sim_colors[1],
                            marker='', ls=ls[j])
            l2[-1][0].set_linestyle(ls[j])

            if j == 0:
                ax[i].annotate(label, xy=(x, 0.93), xycoords='axes fraction',size=12,
                                bbox=dict(boxstyle='round', fc='white', edgecolor='lightgrey'))
                ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
                ax[i].set_ylim(-2.1,ylim)
                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight') 
