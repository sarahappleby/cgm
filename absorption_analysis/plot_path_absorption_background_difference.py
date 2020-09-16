
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
    wave_rest = [1215., 1215., 2796., 1206., 1548., 1031.]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

    # for doing one survey only:
    #cos_survey = ['halos'] * 6
    #cos_survey = ['dwarfs'] * 6
    #lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770' ]
    #plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', r'$\textrm{SiIII}1206$', 
    #              r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    #det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    uvb_labels = [r'$\textrm{HM12} - \textrm{FG20}$', r'$\textrm{HM01} - \textrm{FG20}$']

    model = sys.argv[1]
    wind = sys.argv[2]
    linestyles = ['--', ':']
    markers = ['D', 'v']
    ylim = 0.5
    xoffset = 0.025
    r200_scaled = True
    backgrounds = ['uvb_hm12', 'uvb_hm01']

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+wind+'_background_rho_path_abs_difference'
    #plot_name += '_'+cos_survey[0] +'_only'
    if r200_scaled:
        plot_name += '_scaled'
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(2, 3, figsize=(18.5, 12.5))
    ax = ax.flatten()

    line_hm12 = Line2D([0,1],[0,1],ls=linestyles[0], marker=markers[0], color='grey')
    line_hm01 = Line2D([0,1],[0,1],ls=linestyles[1], marker=markers[1], color='grey')

    leg_uvb = ax[0].legend([line_hm12, line_hm01],uvb_labels, loc=4, fontsize=12)
    ax[0].add_artist(leg_uvb)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=12)
    ax[0].add_artist(leg_color)

    cos_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_obs_path_abs_data.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_obs_path_abs_data.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)

    fg20_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_uvb_fg20_sim_path_abs_data.h5'
    fg20_halos_plot_dict = read_dict_from_h5(fg20_halos_file)
    fg20_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_uvb_fg20_sim_path_abs_data.h5'
    fg20_dwarfs_plot_dict = read_dict_from_h5(fg20_dwarfs_file)

    for b, background in enumerate(backgrounds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_data.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_data.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        if b == 0:
            sim_halos_plot_dict['plot_bins_sf'] -= xoffset
            sim_halos_plot_dict['plot_bins_q'] -= xoffset
            sim_dwarfs_plot_dict['plot_bins_sf'] -= xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] -= xoffset
        elif b == 1:
            sim_halos_plot_dict['plot_bins_sf'] += xoffset
            sim_halos_plot_dict['plot_bins_q'] += xoffset
            sim_dwarfs_plot_dict['plot_bins_sf'] += xoffset
            sim_dwarfs_plot_dict['plot_bins_q'] += xoffset

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                cos_plot_dict = cos_dwarfs_plot_dict
                fg20_plot_dict = fg20_dwarfs_plot_dict
                label = 'COS-Dwarfs'
                x = 0.75
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                cos_plot_dict = cos_halos_plot_dict
                fg20_plot_dict = fg20_halos_plot_dict
                label = 'COS-Halos'
                x = 0.78

            if b == 0:
                ax[i].axhline(0, c='k', ls=':', lw=1)

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

            diff = sim_plot_dict['path_abs_'+lines[i]+'_sf'] - fg20_plot_dict['path_abs_'+lines[i]+'_sf']
            err = np.sqrt(sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf']**2 + fg20_plot_dict['path_abs_'+lines[i]+'_cv_std_sf']**2)
            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], diff,
                            yerr=err,
                            c=sim_colors[0], markersize=6, marker=markers[b], ls=linestyles[b], capsize=4)
            l1[-1][0].set_linestyle(linestyles[b])

            empty_mask = ~np.isnan(sim_plot_dict['path_abs_'+lines[i]+'_q'])
            diff = sim_plot_dict['path_abs_'+lines[i]+'_q'] - fg20_plot_dict['path_abs_'+lines[i]+'_q']
            err = np.sqrt(sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q']**2 + fg20_plot_dict['path_abs_'+lines[i]+'_cv_std_q']**2)
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'][empty_mask], diff[empty_mask],
                            yerr=err[empty_mask],
                            c=sim_colors[1], markersize=6, marker=markers[b], ls=linestyles[b], capsize=4)
            l2[-1][0].set_linestyle(linestyles[b])
           
            ax[i].annotate(label, xy=(x, 0.91), xycoords='axes fraction',size=12,
                                bbox=dict(boxstyle='round', fc='white', edgecolor='lightgrey'))
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(r'${\rm log}\ ({\rm dEW}/ {\rm d} z)\ - {\rm log}\ ({\rm dEW}/ {\rm d} z)_{\rm FG20},\ $' + plot_lines[i], labelpad=0)
            ax[i].set_ylim(-0.8, 1.2)
            if r200_scaled:
                ax[i].set_xlim(0, 1.5)
            else:
                ax[i].set_xlim(25, 120)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')
