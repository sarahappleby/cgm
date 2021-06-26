from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np

from analysis_methods import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

if __name__ == '__main__':
    
    # set some parameters
    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    plot_line_x = [0.8, 0.8, 0.72, 0.73, 0.74, 0.74]

    model = sys.argv[1]
    wind = sys.argv[2]
    
    plot_dir = 'plots/'
    r200_scaled = True
    background = 'uvb_fg20'

    cmap = cm.get_cmap('plasma')
    mass_colors = get_equal_colormap_colors(5, cmap)[::-1]
    mass_bin_labels = ['9.0-9.5', '9.5-10.0', '10.0-10.5', '10.5-11.0', '>11.0']
    mass_plot_labels = [r'$9.0 < \textrm{log} (M_{\star} / M_{\odot}) < 9.5$',
                        r'$9.5 < \textrm{log} (M_{\star} / M_{\odot}) < 10.0$',
                        r'$10.0 < \textrm{log} (M_{\star} / M_{\odot}) < 10.5$',
                        r'$10.5 < \textrm{log} (M_{\star} / M_{\odot}) < 11.0$',
                        r'$ \textrm{log} (M_{\star} / M_{\odot}) > 11.0$']


    plot_name = model+'_'+wind +'_'+background+'_rho_path_abs_mass'
    if r200_scaled:
        scale_str = '_scaled'
        plot_name += scale_str
        xlabel = r'$\rho / r_{200}$'
    else:
        xlabel = r'$\rho (\textrm{kpc})$'
    plot_name += '.png'

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/'
    cos_halos_file = basic_dir+'data/cos_halos_obs_path_abs_data_mass'+scale_str+'.h5'
    cos_halos_plot_dict = read_dict_from_h5(cos_halos_file)
    cos_dwarfs_file = basic_dir+'data/cos_dwarfs_obs_path_abs_data_mass'+scale_str+'.h5'
    cos_dwarfs_plot_dict = read_dict_from_h5(cos_dwarfs_file)
    sim_halos_file = basic_dir+'data/cos_halos_'+model+'_'+wind+'_137_'+background+'_sim_path_abs_data_mass'+scale_str+'.h5'
    sim_halos_plot_dict = read_dict_from_h5(sim_halos_file)
    sim_dwarfs_file = basic_dir+'data/cos_dwarfs_'+model+'_'+wind+'_151_'+background+'_sim_path_abs_data_mass'+scale_str+'.h5'
    sim_dwarfs_plot_dict = read_dict_from_h5(sim_dwarfs_file)
    
    fig, ax = plt.subplots(2, 3, figsize=(21, 12.5))
    ax = ax.flatten()

    mass_lines = []
    for m in range(len(mass_bin_labels)):
        mass_lines.append(Line2D([0,1],[0,1],ls='-', color=mass_colors[m]))
    leg_mass = ax[0].legend(mass_lines,mass_plot_labels, loc=3, fontsize=13, framealpha=0.)
    ax[0].add_artist(leg_mass)

    for i, survey in enumerate(cos_survey):

        # choose the survey and some params
        if survey == 'dwarfs':
            cos_plot_dict = cos_dwarfs_plot_dict.copy()
            sim_plot_dict = sim_dwarfs_plot_dict.copy()
            label = 'COS-Dwarfs'
            cos_marker = '^'
            mass_labels = mass_bin_labels[:3]
            colors = mass_colors[:3]
        elif survey == 'halos':
            cos_plot_dict = cos_halos_plot_dict.copy()
            sim_plot_dict = sim_halos_plot_dict.copy()
            label = 'COS-Halos'
            cos_marker = 'o'
            mass_labels = mass_bin_labels[2:]
            colors = mass_colors[2:]

        data_lines = []
        data_lines.append(Line2D([-1,-1],[0,0], ls='', marker=cos_marker, color='grey', mfc='white', markersize=8))
        data_lines.append(Line2D([-1,-1],[0,0],ls='-', color='grey'))
        leg_data = ax[i].legend(data_lines, [label, 'Simba'], loc=4, fontsize=13, framealpha=0.)
        ax[i].add_artist(leg_data)

        for m, mass_label in enumerate(mass_labels):

            l1, = ax[i].plot(sim_plot_dict['plot_bins'], sim_plot_dict[f'path_abs_{lines[i]}_{mass_label}'], c=colors[m], ls='-', marker='', lw=2)
            ax[i].fill_between(sim_plot_dict['plot_bins'],
                               sim_plot_dict[f'path_abs_{lines[i]}_{mass_label}'] - sim_plot_dict[f'path_abs_{lines[i]}_cv_std_{mass_label}'],
                               sim_plot_dict[f'path_abs_{lines[i]}_{mass_label}'] + sim_plot_dict[f'path_abs_{lines[i]}_cv_std_{mass_label}'],
                               color=colors[m], alpha=0.25)

            c1 = ax[i].errorbar(cos_plot_dict['plot_bins'], cos_plot_dict[f'path_abs_{lines[i]}_{mass_label}'], 
                                yerr=cos_plot_dict[f'path_abs_{lines[i]}_std_{mass_label}'], xerr=cos_plot_dict['xerr'], 
                                capsize=4, c=colors[m], mec=colors[m], mfc='white', marker=cos_marker, markersize=8, ls='')
            for c in range(2):
                c1[-1][c].set_alpha(alpha=0.5)

        ax[i].annotate(plot_lines[i], xy=(plot_line_x[i], 0.88), xycoords='axes fraction',
                        bbox=dict(boxstyle='round', fc='white'))

        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z),\ $' + plot_lines[i])       
        ax[i].set_ylim(0.8, 3.)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name, bbox_inches = 'tight')



