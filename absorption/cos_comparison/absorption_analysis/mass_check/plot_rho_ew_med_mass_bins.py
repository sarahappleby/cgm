import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    # set some parameters
    cos_survey = ['halos'] * 4
    lines = ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{OVI}1031$']
    det_thresh = np.log10([0.2, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

    model = sys.argv[1]
    wind = sys.argv[2]
    
    plot_dir = 'mass_check/'
    r200_scaled = True
    ylim = 0.7
    background = 'uvb_hm12'
    markers = []
    linestyles = ['--', ':']
    bin_labels = ['10.0-10.5', '10.5-11.0']

    sim_colors, cos_colors = get_tol_colors()

    # set plot name according to parameters
    plot_name = model+'_'+wind +'_'+background+'_rho_ew_med_mass_bins'
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
    sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_ew_med_data.h5'
    sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)

    fig, ax = plt.subplots(2, 2, figsize=(12, 14))
    ax = ax.flatten()

    line_sf = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[0])
    line_q = Line2D([0,1],[0,1],ls='-', marker=None, color=sim_colors[1])

    leg_color = ax[0].legend([line_sf, line_q],['Simba SF', 'Simba Q'], loc=3, fontsize=12)
    ax[0].add_artist(leg_color)

    line_b1 = Line2D([0,1],[0,1],ls=linestyles[0], color='grey')
    line_b2 = Line2D([0,1],[0,1],ls=linestyles[1], color='grey')

    leg_bins = ax[0].legend([line_b1, line_b2],bin_labels, loc=4, fontsize=12)
    ax[0].add_artist(leg_bins)

    for i, survey in enumerate(cos_survey):

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


        for b, bin_label in enumerate(bin_labels):

            # plot the Simba data as lines
            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf_'+bin_label], sim_plot_dict['EW_'+lines[i]+'_med_sf_'+bin_label], 
                            yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_sf_'+bin_label], 
                            capsize=4, c=sim_colors[0], marker='o', ls=linestyles[b])
            l1[-1][0].set_linestyle(linestyles[b])
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q_'+bin_label], sim_plot_dict['EW_'+lines[i]+'_med_q_'+bin_label], 
                            yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_q_'+bin_label], 
                            capsize=4, c=sim_colors[1], marker='o', ls=linestyles[b])
            l2[-1][0].set_linestyle(linestyles[b])
            
            if b == 0:
                ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
                ax[i].set_ylim(-2, ylim)
                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name)
