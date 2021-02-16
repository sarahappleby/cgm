
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=17)

if __name__ == '__main__':


    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    wave_rest = [1215., 1215., 2796., 1206., 1548., 1031.]
    plot_line_x = [0.78, 0.78, 0.72, 0.73, 0.74, 0.74]
    plot_lines = [r'$\textrm{HI}1215$', r'$\textrm{HI}1215$', r'$\textrm{MgII}2796$',
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
    lower_lim = 0.5

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+background+'_winds_rho_path_abs_thresh'
    if r200_scaled:
        scale_str = '_scaled'
        plot_name += scale_str
        xlabel = r'$\rho / r_{200}$'
    else:
        scale_str = '' 
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharey='row', sharex='col')
    ax = ax.flatten()

    wind_lines = []
    for w in rage(len(winds)):
        wind_lines.append(Line2D([0,1],[0,1],ls=ls[w], color='grey'))
    leg_winds = ax[0].legend(wind_lines,wind_labels, loc=4, fontsize=15, framealpha=0.)
    ax[0].add_artist(leg_winds)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=15, framealpha=0.)
    ax[0].add_artist(leg_color)

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/' 
    cos_halos_file = basic_dir+'data/cos_halos_obs_path_abs_thresh_data'+scale_str+'.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = basic_dir+'data/cos_dwarfs_obs_path_abs_thresh_data'+scale_str+'.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)

    for j, wind in enumerate(winds):

        sim_halos_file = basic_dir+'data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_thresh_data'+scale_str+'.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                cos_plot_dict = cos_dwarfs_plot_dict
                label = 'COS-Dwarfs'
                x = 0.72
                cos_marker = '^'
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                cos_plot_dict = cos_halos_plot_dict
                label = 'COS-Halos'
                x = 0.75
                cos_marker = 'o'

            if j == 0:

                l1, = ax[i].plot(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'], c=sim_colors[0], ls='-', marker='', lw=2)
                ax[i].fill_between(sim_plot_dict['plot_bins_sf'],
                                   sim_plot_dict['path_abs_'+lines[i]+'_sf'] - sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf'],
                                   sim_plot_dict['path_abs_'+lines[i]+'_sf'] + sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf'],
                                   color=sim_colors[0], alpha=0.25)
                l2, = ax[i].plot(sim_plot_dict['plot_bins_q'], sim_plot_dict['path_abs_'+lines[i]+'_q'], c=sim_colors[1], ls='-', marker='', lw=2)
                ax[i].fill_between(sim_plot_dict['plot_bins_q'],
                                   sim_plot_dict['path_abs_'+lines[i]+'_q'] - sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q'],
                                   sim_plot_dict['path_abs_'+lines[i]+'_q'] + sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q'],
                                   color=sim_colors[1], alpha=0.25)

                lower_lim_mask = (sim_plot_dict['path_abs_'+lines[i]+'_q'] <= lower_lim)
                lower_lim_array = np.array([lower_lim] * len(lower_lim_mask))
                ax[i].plot(sim_plot_dict['plot_bins_q'][lower_lim_mask], lower_lim_array[lower_lim_mask],
                                        c=sim_colors[1], markersize=15, marker='$\downarrow$', ls='')

            else:

                ax[i].plot(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'],
                            c=sim_colors[0], ls=ls[j], lw=2.)
                if not wind in ['s50nojet', 's50nofb']:
                    ax[i].plot(sim_plot_dict['plot_bins_q'], sim_plot_dict['path_abs_'+lines[i]+'_q'],
                                c=sim_colors[1], ls=ls[j], lw=2.)

            c1 = ax[i].errorbar(cos_plot_dict['plot_bins_sf'], cos_plot_dict['path_abs_'+lines[i]+'_sf'],
                            yerr=cos_plot_dict['path_abs_'+lines[i]+'_std_sf'], xerr=cos_plot_dict['xerr_sf'],
                        capsize=4, c=cos_colors[0], mec=cos_colors[0], mfc='white', marker=cos_marker, markersize=8, ls='')
            c2 = ax[i].errorbar(cos_plot_dict['plot_bins_q'], cos_plot_dict['path_abs_'+lines[i]+'_q'],
                            yerr=cos_plot_dict['path_abs_'+lines[i]+'_std_q'], xerr=cos_plot_dict['xerr_q'],
                            capsize=4, c=cos_colors[1], mec=cos_colors[1], mfc='white', marker=cos_marker, markersize=8, ls='')
            for c in range(2):
                c1[-1][c].set_alpha(alpha=0.5)
                c2[-1][c].set_alpha(alpha=0.5)
            leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=13.5, loc=1, framealpha=0.)

            ax[i].set_xlabel(xlabel)
            ax[i].annotate(plot_lines[i], xy=(plot_line_x[i], 0.73), xycoords='axes fraction',size=15,
                        bbox=dict(boxstyle='round', fc='white'))
            ax[i].set_ylim(0.4, 3.0)
            if r200_scaled:
                ax[i].set_xlim(0., 1.5)
            else:
                ax[i].set_xlim(25, 120)
            if i in [0, 3]:
                ax[i].set_ylabel(r'$\textrm{log}\ \textrm{dEW}/ \textrm{d} z$')
    
    plt.setp(ax[3].get_yticklabels()[-1], visible=False)
    plt.setp(ax[3].get_xticklabels()[-1], visible=False)
    plt.setp(ax[4].get_xticklabels()[-1], visible=False)
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight') 
