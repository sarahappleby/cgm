import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from plot_cos_data import plot_dwarfs_civ, plot_dwarfs_lya, plot_halos
from physics import median_ew_cos_groups, convert_to_log

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', 
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon

    model = 'm100n1024'
    wind = 's50'
    mlim = np.log10(5.8e8) # lower limit of M*
    ylim = 0.7
    plot_dir = 'plots/'
    h = 0.68
    r200_scaled = False

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    plot_name = model+'_'+wind +'_rho_cfrac'
    if r200_scaled:
        plot_name += '_scaled'
    plot_name += '.png'

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    halo_rho, halo_M, halo_r200, halo_ssfr = get_cos_halos()
    dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr = get_cos_dwarfs()


    for i, survey in enumerate(cos_survey):

        data_dict = {}
        cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
        with h5py.File(cos_sample_file, 'r') as f:
            data_dict['mass'] = np.repeat(f['mass'][:], 4)
            data_dict['ssfr'] = np.repeat(f['ssfr'][:], 4)
            data_dict['pos'] = np.repeat(f['position'][:], 4, axis=0)
            data_dict['r200'] = np.repeat(f['halo_r200'][:], 4)
        data_dict['ssfr'][data_dict['ssfr'] < -11.5] = -11.5

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

        mass_mask = cos_dict['M'] > mlim
        for k in cos_dict.keys():
            cos_dict[k] = cos_dict[k][mass_mask]

        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            mass_mask = np.delete(mass_mask, 3)
            for k in cos_dict.keys():
                cos_dict[k] = np.delete(cos_dict[k], 3)
        
        cos_rho_long = np.repeat(cos_dict['rho'], 20)

        ew_file = 'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_ew_data_lsf.h5'
        with h5py.File(ew_file, 'r') as f:
            data_dict['ew'] = f[lines[i]+'_wave_ew'][:]

        # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            for k in data_dict.keys():
                data_dict[k] = np.delete(data_dict[k], np.arange(3*20, 4*20), axis=0)


        if r200_scaled:
            cos_dict['cos_dist'] = cos_dict['rho'] / cos_dict['r200']
            data_dict['sim_dist'] = cos_rho_long / data_dict['r200']
            xlabel = r'$\rho / r_{200}$'
        else:
            cos_dict['cos_dist'] = cos_dict['rho'].copy()
            data_dict['sim_dist'] = cos_rho_long.copy()
            xlabel = r'$\rho (\textrm{kpc})$'

        data_dict['ew'], data_dict['ew_err'], data_dict['median_r'], data_dict['median_ssfr'] = \
                median_ew_cos_groups(data_dict['ew'], data_dict['sim_dist'], data_dict['ssfr'], 20, len(cos_dict['rho'])) 

        mask = data_dict['median_ssfr'] < quench
        l1 = ax[i].errorbar(data_dict['median_r'][mask], data_dict['ew'][mask], yerr=[data_dict['ew_err'][0][mask], data_dict['ew_err'][1][mask]], \
                            ms=3.5, marker='s', capsize=4, ls='', c='r')
        mask = data_dict['median_ssfr'] > quench
        l2 = ax[i].errorbar(data_dict['median_r'][mask], data_dict['ew'][mask], yerr=[data_dict['ew_err'][0][mask], data_dict['ew_err'][1][mask]], \
                ms=3.5, marker='s', capsize=4, ls='', c='b')
        if i == 0:
            leg1 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], fontsize=10.5, loc=4)

        ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
        ax[i].set_ylim(-2.5, ylim)
        if r200_scaled:
            ax[i].set_xlim(0, 1.5)
        else:
            ax[i].set_xlim(25, 145)

        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            c1, c2 = plot_dwarfs_civ(ax[i], quench, r200_scaled=r200_scaled)
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            c1, c2 = plot_dwarfs_lya(ax[i], quench, r200_scaled=r200_scaled)
        elif (survey == 'halos'):
            c1, c2 = plot_halos(ax[i], lines[i], quench, r200_scaled)

        leg2 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], loc=3, fontsize=10.5)

        if i == 0:
            ax[i].add_artist(leg1)

    plt.savefig(plot_dir+plot_name)


