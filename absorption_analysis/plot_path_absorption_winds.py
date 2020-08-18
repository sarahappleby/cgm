
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

    model = 'm50n512'
    winds = ['s50j7k', 's50nox', 's50nojet', 's50noagn']
    ls = ['-', '--', (0, (3, 5, 1, 5, 1, 5)), ':']
    markers = ['o', 'D', 's', 'v']
    ylim = 0.5
    r200_scaled = True

    plot_dir = 'plots/'
    plot_name = model+'_winds_rho_path_abs'
    if r200_scaled:
        plot_name += '_scaled'
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    line_sim = Line2D([0,1],[0,1],ls=ls[0], color='k')
    line_jet = Line2D([0,1],[0,1],ls=ls[1], color='k')
    line_x = Line2D([0,1],[0,1],ls=ls[2], color='k')
    line_agn = Line2D([0,1],[0,1],ls=ls[3], color='k')

    leg = ax[1].legend([line_sim, line_jet, line_x, line_agn],winds, loc=1, fontsize=12)
    ax[1].add_artist(leg)

    for j, wind in enumerate(winds):

        sim_halos_file = '/home/sapple/cgm/absorption_analysis/data/cos_halos_'+model+'_'+wind+'_137_sim_path_abs_data.h5'
        sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
        sim_dwarfs_file = '/home/sapple/cgm/absorption_analysis/data/cos_dwarfs_'+model+'_'+wind+'_151_sim_path_abs_data.h5'
        sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)

        for i, survey in enumerate(cos_survey):

            # choose the survey and some params
            if survey == 'dwarfs':
                sim_plot_dict = sim_dwarfs_plot_dict
                label = 'COS-Dwarfs'
            elif survey == 'halos':
                sim_plot_dict = sim_halos_plot_dict
                label = 'COS-Halos'

            l1 = ax[i].errorbar(sim_plot_dict['plot_bins_sf'], sim_plot_dict['path_abs_'+lines[i]+'_sf'],
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_sf'],
                            c='b', markersize=6, marker=markers[j], ls=ls[j])
            l2 = ax[i].errorbar(sim_plot_dict['plot_bins_q'], sim_plot_dict['path_abs_'+lines[i]+'_q'],
                            yerr=sim_plot_dict['path_abs_'+lines[i]+'_cv_std_q'],
                            c='r', markersize=6, marker=markers[j], ls=ls[j])
            if i == 0:
                leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z)\ $' + plot_lines[i])
            ax[i].set_ylim(1.5, 3.0)
            if r200_scaled:
                ax[i].set_xlim(0, 1.5)
            else:
                ax[i].set_xlim(25, 120)

    plt.savefig(plot_dir+plot_name) 
