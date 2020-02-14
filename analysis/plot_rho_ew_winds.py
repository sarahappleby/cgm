import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from plot_cos_data import plot_dwarfs_civ, plot_dwarfs_lya, plot_halos
from physics import do_exclude_outliers, sim_binned_ew, do_bins, median_ew_cos_groups

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = sys.argv[1]

    cos_survey = [cos_survey] * 6
    lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                        r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

    model = 'm50n512'
    winds = ['s50j7k', 's50nojet', 's50nox', 's50noagn']
    ls = ['-', '--', '-.', ':']
    nbins = 4
    out = 5.
    mlim = np.log10(5.8e8) # lower limit of M*
    ylim = 0.5
    h = 0.68
    r200_scaled = True
    do_equal_bins = False

    plot_dir = 'plots/'
    plot_name = model+'_winds_rho_ew'
    if r200_scaled:
        plot_name += '_scaled'
    plot_name += '.png'

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    if do_equal_bins:
        if r200_scaled:
            r_end = 1.
            dr = .2
        else:
            r_end = 200.
            dr = 40.
        rho_bins_sim_q = np.arange(0., r_end, dr)
        rho_bins_cos_q = np.arange(0., r_end, dr)
        rho_bins_sim_sf = np.arange(0., r_end, dr)
        rho_bins_cos_sf = np.arange(0., r_end, dr)
        plot_bins_sim_q = rho_bins_sim[:-1] + 0.5*dr
        plot_bins_cos_q = rho_bins_cos[:-1] + 0.5*dr
        plot_bins_sim_sf = rho_bins_sim[:-1] + 0.5*dr
        plot_bins_cos_sf = rho_bins_cos[:-1] + 0.5*dr

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    line_sim = Line2D([0,1],[0,1],ls=ls[0], color='k')
    line_jet = Line2D([0,1],[0,1],ls=ls[1], color='k')
    line_x = Line2D([0,1],[0,1],ls=ls[2], color='k')
    line_agn = Line2D([0,1],[0,1],ls=ls[3], color='k')

    leg = ax[1].legend([line_sim, line_jet, line_x, line_agn],winds, loc=1, fontsize=12)
    ax[1].add_artist(leg)

    halo_rho, halo_M, halo_r200, halo_ssfr = get_cos_halos()
    dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr = get_cos_dwarfs()

    for i, survey in enumerate(cos_survey):
        
        cos_dict = {}
        if survey == 'dwarfs':
            label = 'COS-Dwarfs'
            snap = '151'
            z = 0.
            cos_dict['rho'], cos_dict['M'], cos_dict['r200'], cos_dict['ssfr'] = dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr
        elif survey == 'halos':
            label = 'COS-Halos'
            snap = '137'
            z = 0.2
            cos_dict['rho'], cos_dict['M'], cos_dict['r200'], cos_dict['ssfr'] = halo_rho, halo_M, halo_r200, halo_ssfr
        quench = -1.8  + 0.3*z - 9.

        if r200_scaled:
            cos_dict['rho'] = cos_dict['rho'].astype(float)
            cos_dict['rho'] *= h * (1+z) # get in kpc/h
    
        mass_mask = cos_dict['M'] > mlim
        for k in cos_dict.keys():
            cos_dict[k] = cos_dict[k][mass_mask]

        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            mass_mask = np.delete(mass_mask, 3)
            for k in cos_dict.keys():
                cos_dict[k] = np.delete(cos_dict[k], 3)

        cos_rho_long = np.repeat(cos_dict['rho'], 20)
        
        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            c1, c2 = plot_dwarfs_civ(ax[i], quench, r200_scaled=r200_scaled)
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            c1, c2 = plot_dwarfs_lya(ax[i], quench, r200_scaled=r200_scaled)
        elif (survey == 'halos'):
            c1, c2 = plot_halos(ax[i], lines[i], quench, r200_scaled)

        leg2 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], loc=3, fontsize=10.5)
        if i == 1:
            ax[i].add_artist(leg)

        for j, w in enumerate(winds):

            data_dict = {}
            cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+w+'_cos_'+survey+'_sample.h5'
            with h5py.File(cos_sample_file, 'r') as f:
                data_dict['mass'] = np.repeat(f['mass'][:], 4)
                data_dict['ssfr'] = np.repeat(f['ssfr'][:], 4)
                data_dict['pos'] = np.repeat(f['position'][:], 4, axis=0)
                data_dict['r200'] = np.repeat(f['halo_r200'][:], 4)
            data_dict['ssfr'][data_dict['ssfr'] < -11.5] = -11.5

            ew_file = 'data/cos_'+survey+'_'+model+'_'+w+'_'+snap+'_ew_data_lsf.h5'
            with h5py.File(ew_file, 'r') as f:
                data_dict['ew'] = f[lines[i]+'_wave_ew'][:]

            if (survey == 'dwarfs') & (lines[i] == 'H1215'):
                for k in data_dict.keys():
                    data_dict[k] = np.delete(data_dict[k], np.arange(3*20, 4*20), axis=0)
            
            if r200_scaled:
                data_dict['sim_dist'] = cos_rho_long / data_dict['r200']
                xlabel = r'$\rho / r_{200}$'
            else:
                data_dict['sim_dist'] = cos_rho_long.copy()
                xlabel = r'$\rho (\textrm{kpc})$'

            if not do_equal_bins:
                data_dict = do_exclude_outliers(data_dict, out)
                mask = (data_dict['ssfr'] > quench)
                rho_bins_sim_sf, plot_bins_sim_sf = do_bins(data_dict['sim_dist'][mask], nbins)
                mask = (data_dict['ssfr'] < quench)
                rho_bins_sim_q, plot_bins_sim_q = do_bins(data_dict['sim_dist'][mask], nbins)

            mask = data_dict['ssfr'] > quench
            sf_ew, sf_err = sim_binned_ew(data_dict, mask, rho_bins_sim_sf, boxsize)
            mask = data_dict['ssfr'] < quench
            q_ew, q_err = sim_binned_ew(data_dict, mask, rho_bins_sim_q, boxsize)
            
            l1, = ax[i].plot(plot_bins_sim_sf, sf_ew, linestyle=ls[j], c='b', label='Simba SF')
            l2, = ax[i].plot(plot_bins_sim_q, q_ew, linestyle=ls[j], c='r', label='Simba Q')
            if j == 0:
                if i == 1:
                    leg3 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], fontsize=10.5, loc=2)
                    ax[i].add_artist(leg)
                ax[i].fill_between(plot_bins_sim_sf, sf_ew - sf_err, sf_ew + sf_err, linestyle=ls[j], color='b', alpha=0.1)
                ax[i].fill_between(plot_bins_sim_q, q_ew - q_err, q_ew + q_err, linestyle=ls[j], color='r', alpha=0.1)


        ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
        ax[i].set_ylim(-2.,ylim)
        if r200_scaled:
            ax[i].set_xlim(0, 1.1)
        else:
            ax[i].set_xlim(25, 145)


    plt.savefig(plot_dir+plot_name) 
