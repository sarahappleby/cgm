
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
    uvb_labels = [r'$\textrm{FG20}$', r'$\textrm{HM12 x2}$', r'$\textrm{HM01}$']

    # for doing one survey only:
    #cos_survey = ['halos'] * 6
    #cos_survey = ['dwarfs'] * 6
    #lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770' ]
    #plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', r'$\textrm{SiIII}1206$', 
    #              r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    #det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1])

    model = sys.argv[1]
    wind = sys.argv[2]
    linestyles = ['-', '--', ':']
    ylim = 0.5
    xoffset = 0.025
    r200_scaled = True
    backgrounds = ['uvb_fg20', 'uvb_hm12_x2', 'uvb_hm01']

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_'+wind+'_background_cfrac'
    #plot_name += '_'+cos_survey[0] +'_only'
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

    line_fg20 = Line2D([0,1],[0,1],ls=linestyles[0], color='grey')
    line_hm12_x2 = Line2D([0,1],[0,1],ls=linestyles[1], color='grey')
    line_hm01 = Line2D([0,1],[0,1],ls=linestyles[2], color='grey')

    leg_uvb = ax[0].legend([line_fg20, line_hm12_x2, line_hm01],uvb_labels, loc=4, fontsize=14, framealpha=0.)
    ax[0].add_artist(leg_uvb)

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=14, framealpha=0.)
    ax[0].add_artist(leg_color)

    cos_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_obs_cfrac_data'+scale_str+'.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_obs_cfrac_data'+scale_str+'.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)

    for b, background in enumerate(backgrounds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_cfrac_data'+scale_str+'.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_cfrac_data'+scale_str+'.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict.copy()
                cos_plot_dict = cos_dwarfs_plot_dict.copy()
                label = 'COS-Dwarfs'
                x = 0.75
                cos_marker = '^'
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict.copy()
                cos_plot_dict = cos_halos_plot_dict.copy()
                label = 'COS-Halos'
                x = 0.77
                cos_marker = 'o'
            
            if (b == 2) & ('cfrac_'+lines[i]+'_sf' in list(cos_plot_dict.keys())):
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

            if b == 0:

                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel(r'$f_\textrm{cov},\ $' + plot_lines[i])
                ax[i].set_ylim(0, 1.1)
                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 145)

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

            else:

                l1 = ax[i].plot(sim_plot_dict['plot_bins_sf'], sim_plot_dict['cfrac_'+lines[i]+'_sf'],
                                color=sim_colors[0], lw=2, ls=linestyles[b])
                empty_mask = ~np.isnan(sim_plot_dict['cfrac_'+lines[i]+'_q'])
                l2 = ax[i].plot(sim_plot_dict['plot_bins_q'][empty_mask], sim_plot_dict['cfrac_'+lines[i]+'_q'][empty_mask],
                                color=sim_colors[1], lw=2, ls=linestyles[b])

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')
