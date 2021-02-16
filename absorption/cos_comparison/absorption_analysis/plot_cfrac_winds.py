
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
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031',]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

    model = 'm50n512'
    winds = ['s50j7k', 's50nox', 's50nojet', 's50nofb']
    wind_labels = [r'$\textrm{Simba}$', r'$\textrm{No-Xray}$', r'$\textrm{No-jet}$', r'$\textrm{No-feedback}$']
    ls = ['-', '--', ':', '-.']
    markers = ['o', 'D', 's', '^']
    ylim = 0.5
    xoffset = 0.035
    r200_scaled = True
    background = 'uvb_fg20'
    lower_lim = 0.

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+background+'_winds_rho_cfrac'
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

    wind_lines = []
    for w in range(len(winds)):
        wind_lines.append(Line2D([0,1],[0,1],ls=ls[w], color='grey'))
    leg_winds = ax[0].legend(wind_lines,wind_labels, loc=4, fontsize=16, framealpha=0.)
    ax[0].add_artist(leg_winds)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])
    
    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=16, framealpha=0.)
    ax[0].add_artist(leg_color)

    for j, wind in enumerate(winds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_cfrac_data'+scale_str+'.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_cfrac_data'+scale_str+'.h5'
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
                sim_plot_dict = sim_dwarfs_plot_dict.copy()
                label = 'COS-Dwarfs'
                x = 0.72
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict.copy()
                label = 'COS-Halos'
                x = 0.75
    
            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['cfrac_'+lines[i]+'_sf'],
                            yerr=sim_plot_dict['cfrac_'+lines[i]+'_poisson_sf'], capsize=4, c=sim_colors[0],
                            markersize=6, marker=markers[j], ls=ls[j])
            l1[-1][0].set_linestyle(ls[j])

            empty_mask = ~np.isnan(sim_plot_dict['cfrac_'+lines[i]+'_q'])
            lower_lim_array = np.array([lower_lim] * len(empty_mask))

            ax[i].plot(sim_plot_dict['plot_bins_q'][empty_mask], sim_plot_dict['cfrac_'+lines[i]+'_q'][empty_mask],
                            c=sim_colors[1], markersize=6, marker=markers[j], ls='')
            ax[i].plot(sim_plot_dict['plot_bins_q'][~empty_mask], lower_lim_array[~empty_mask],
                            c=sim_colors[1], markersize=15, marker='$\downarrow$', ls='')
            sim_plot_dict['cfrac_'+lines[i]+'_q'][~empty_mask] = lower_lim
            sim_plot_dict['cfrac_'+lines[i]+'_poisson_q'][~empty_mask] = np.nan
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['cfrac_'+lines[i]+'_q'],
                            yerr=sim_plot_dict['cfrac_'+lines[i]+'_poisson_q'], capsize=4, c=sim_colors[1],
                            marker='', ls=ls[j])
            l2[-1][0].set_linestyle(ls[j])


            if j == 0:
                ax[i].annotate(label, xy=(x, 0.93), xycoords='axes fraction',size=16,
                                bbox=dict(boxstyle='round', fc='none', edgecolor='none'))
                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel(r'$f_\textrm{cov},\ $' + plot_lines[i])
                ax[i].set_ylim(-0.1, 1.1)

                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight') 
