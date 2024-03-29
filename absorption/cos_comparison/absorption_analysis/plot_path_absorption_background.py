
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
    plot_lines = [r'$\textrm{HI}1215$', r'$\textrm{HI}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    plot_line_x = [0.78, 0.78, 0.72, 0.73, 0.74, 0.74]
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

    # for doing one survey only:
    #cos_survey = ['halos'] * 6
    #cos_survey = ['dwarfs'] * 6
    #lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770' ]
    #plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', r'$\textrm{SiIII}1206$', 
    #              r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    #det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    uvb_labels = [r'$\textrm{FG20}$', r'$\textrm{HM12 x2}$', r'$\textrm{HM01}$']

    model = sys.argv[1]
    wind = sys.argv[2]
    linestyles = ['-', '--', ':']
    ylim = 0.5
    xoffset = 0.025
    r200_scaled = True
    backgrounds = ['uvb_fg20', 'uvb_hm12_x2', 'uvb_hm01']

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+wind+'_background_rho_path_abs'
    #plot_name += '_'+cos_survey[0] +'_only'
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

    line_fg20 = Line2D([0,1],[0,1],ls=linestyles[0], color='grey')
    line_hm12_x2 = Line2D([0,1],[0,1],ls=linestyles[1], color='grey')
    line_hm01 = Line2D([0,1],[0,1],ls=linestyles[2], color='grey')

    leg_uvb = ax[0].legend([line_fg20, line_hm12_x2, line_hm01],uvb_labels, loc=4, fontsize=15, framealpha=0.)
    ax[0].add_artist(leg_uvb)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=15, framealpha=0.)
    ax[0].add_artist(leg_color)

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/'
    cos_halos_file = basic_dir+'data/cos_halos_obs_path_abs_thresh_data'+scale_str+'.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = basic_dir+'data/cos_dwarfs_obs_path_abs_thresh_data'+scale_str+'.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)

    for b, background in enumerate(backgrounds):

        sim_halos_file = basic_dir+'data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_data'+scale_str+'.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_data'+scale_str+'.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        for i, survey in enumerate(cos_survey):

            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                cos_plot_dict = cos_dwarfs_plot_dict
                label = 'COS-Dwarfs'
                x = 0.75
                cos_marker = '^'
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                cos_plot_dict = cos_halos_plot_dict
                label = 'COS-Halos'
                x = 0.77
                cos_marker = 'o'

            if (b == 2) & ('path_abs_'+lines[i]+'_sf' in list(cos_plot_dict.keys())):
                c1 = ax[i].errorbar(cos_plot_dict['plot_bins_sf'], cos_plot_dict['path_abs_'+lines[i]+'_sf'],
                                yerr=cos_plot_dict['path_abs_'+lines[i]+'_quad_sf'], xerr=cos_plot_dict['xerr_sf'],
                                capsize=4, c=cos_colors[0], mec=cos_colors[0], mfc='white', marker=cos_marker, markersize=8, ls='')
                c2 = ax[i].errorbar(cos_plot_dict['plot_bins_q'], cos_plot_dict['path_abs_'+lines[i]+'_q'],
                                yerr=cos_plot_dict['path_abs_'+lines[i]+'_quad_q'], xerr=cos_plot_dict['xerr_q'],
                                capsize=4, c=cos_colors[1], mec=cos_colors[1], mfc='white', marker=cos_marker, markersize=8, ls='')
                for c in range(2):
                    c1[-1][c].set_alpha(alpha=0.5)
                    c2[-1][c].set_alpha(alpha=0.5)
                #c1[-1][0].set_linestyle('--')
                #c2[-1][0].set_linestyle('--')
                leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=15, loc=1, framealpha=0.)

            if b == 0:
    
                ax[i].set_xlabel(xlabel)
                ax[i].annotate(plot_lines[i], xy=(plot_line_x[i], 0.73), xycoords='axes fraction',
                        bbox=dict(boxstyle='round', fc='white'))
                ax[i].set_ylim(0.9, 3.0)
                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 120)
                if i in [0, 3]:
                    ax[i].set_ylabel(r'$\textrm{log}\ \textrm{dEW}/ \textrm{d} z$')

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

            else:

                l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'],
                                    c=sim_colors[0], lw=2, ls=linestyles[b])
                empty_mask = ~np.isnan(sim_plot_dict['path_abs_'+lines[i]+'_q'])
                l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'][empty_mask], sim_plot_dict['path_abs_'+lines[i]+'_q'][empty_mask],
                                    c=sim_colors[1], lw=2, ls=linestyles[b])

    plt.setp(ax[3].get_xticklabels()[-1], visible=False)
    plt.setp(ax[4].get_xticklabels()[-1], visible=False)
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')
