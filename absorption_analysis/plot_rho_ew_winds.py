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
    winds = ['s50j7k', 's50nox', 's50nojet', 's50noagn']
    wind_labels = [r'$\textrm{Simba}$', r'$\textrm{No-Xray}$', r'$\textrm{No-jet}$', r'$\textrm{No-AGN}$']
    ls = ['-', '--', (0, (3, 5, 1, 5, 1, 5)), ':']
    markers = ['o', 'D', 's', 'v']
    ylim = 0.5
    r200_scaled = True

    sim_colors, cos_colors = get_tol_colors()

    plot_dir = 'plots/'
    plot_name = model+'_winds_rho_ew'
    if r200_scaled:
        plot_name += '_scaled'
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    line_sim = Line2D([0,1],[0,1],ls=ls[0], marker=markers[0], color='grey')
    line_jet = Line2D([0,1],[0,1],ls=ls[1], marker=markers[1], color='grey')
    line_x = Line2D([0,1],[0,1],ls=ls[2], marker=markers[2], color='grey')
    line_agn = Line2D([0,1],[0,1],ls=ls[3], marker=markers[3], color='grey')

    leg = ax[0].legend([line_sim, line_jet, line_x, line_agn],wind_labels, loc=1, fontsize=12)
    ax[0].add_artist(leg)

    for j, wind in enumerate(winds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_sim_ew_med_data.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_sim_ew_med_data.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)


        for i, survey in enumerate(cos_survey):
        
            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                label = 'COS-Dwarfs'
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                label = 'COS-Halos'

            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['EW_'+lines[i]+'_med_sf'], 
                                yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_sf'], 
                                capsize=4, c=sim_colors[0], markersize=6, marker=markers[j], linestyle=ls[j], label='Simba SF')
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['EW_'+lines[i]+'_med_q'], 
                                yerr=sim_plot_dict['EW_'+lines[i]+'_cosmic_std_q'], 
                                capsize=4, c=sim_colors[1], markersize=6, marker=markers[j], linestyle=ls[j], label='Simba Q')
            if j == 0:
                if i == 0:
                    leg3 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], fontsize=10.5, loc=3)

            if j == 0:
                ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
                ax[i].set_xlabel(xlabel)
                ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
                ax[i].set_ylim(-2.,ylim)
                if r200_scaled:
                    ax[i].set_xlim(0, 1.5)
                else:
                    ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name) 
