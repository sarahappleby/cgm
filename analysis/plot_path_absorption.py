
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from physics import do_bins, do_exclude_outliers, compute_path_length, sim_binned_path_abs, cos_binned_path_abs

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    wave_rest = [1215., 1215., 2796., 1206., 1548., 1031.]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    model = 'm100n1024'
    wind = 's50'
    velocity_width = 300. # km/s
    mlim = np.log10(5.8e8) # lower limit of M*
    plot_dir = 'plots/'
    r200_scaled = True
    do_equal_bins = False
    nbins_sim = 4
    nbins_cos = 3
    out = 5.
    h = 0.68

    plot_name = model+'_'+wind +'_rho_path_abs'
    if r200_scaled:
        plot_name += '_scaled'
    if do_equal_bins:
        plot_name += '_equal_bins'
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

    halo_rho, halo_M, halo_r200, halo_ssfr = get_cos_halos()
    dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr = get_cos_dwarfs()
    
    for i, survey in enumerate(cos_survey):

        data_dict = {}
        cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
        with h5py.File(cos_sample_file, 'r') as f:
            keys = list(f.keys())
            if 'path_length_'+lines[i] not in keys:
                vgal = f['vgal_position'][:][:, 2]
                path_length = compute_path_length(vgal, velocity_width, wave_rest[i], z)
                f.create_dataset('path_length_'+lines[i], data=np.array(path_length))
            
            data_dict['path_length'] = np.repeat(f['path_length_'+lines[i]][:], 4)
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

        if not do_equal_bins:
            data_dict = do_exclude_outliers(data_dict, out)
            mask = (data_dict['ssfr'] > quench)
            rho_bins_sim_sf, plot_bins_sim_sf = do_bins(data_dict['sim_dist'][mask], nbins_sim)
            mask = (data_dict['ssfr'] < quench)
            rho_bins_sim_q, plot_bins_sim_q = do_bins(data_dict['sim_dist'][mask], nbins_sim)
            mask = (cos_dict['ssfr'] > quench)
            rho_bins_cos_sf, plot_bins_cos_sf = do_bins(cos_dict['cos_dist'][mask], nbins_cos)
            mask = (cos_dict['ssfr'] < quench)
            rho_bins_cos_q, plot_bins_cos_q = do_bins(cos_dict['cos_dist'][mask], nbins_cos)


        sim_sf_path_abs, sim_sf_err = sim_binned_path_abs(data_dict, (data_dict['ssfr'] > quench), rho_bins_sim_sf, det_thresh[i], boxsize)
        sim_q_path_abs, sim_q_err = sim_binned_path_abs(data_dict, (data_dict['ssfr'] < quench), rho_bins_sim_q, det_thresh[i], boxsize)

        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            cos_dict['EW'], cos_dict['EWerr'], cos_dict['EW_less_than'] = get_cos_dwarfs_civ() #in mA
            cos_dict['EW'] /= 1000.
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            cos_dict['EW'], cos_dict['EWerr'] = get_cos_dwarfs_lya() # in mA
            cos_dict['EW'] /= 1000.
            cos_dict['EW'] = np.delete(cos_dict['EW'], 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
        elif (survey == 'halos'):
            cos_dict['EW'], cos_dict['EWerr'] = read_halos_data(lines[i])
            cos_dict['EW'] = np.abs(cos_dict['EW'])
        cos_dict['EW'] = cos_dict['EW'][mass_mask]

        if survey == 'halos':
            ew_mask = cos_dict['EW'] > 0.
            for k in cos_dict.keys():
                cos_dict[k] = cos_dict[k][ew_mask]
        
        cos_dict['path_length'] = np.repeat(data_dict['path_length'][0], len(cos_dict['EW']))
        
        cos_sf_path_abs = cos_binned_path_abs(cos_dict, (cos_dict['ssfr'] > quench), rho_bins_cos_sf, det_thresh[i])
        cos_q_path_abs = cos_binned_path_abs(cos_dict, (cos_dict['ssfr'] < quench), rho_bins_cos_q, det_thresh[i])
       
        cos_sf_path_abs[cos_sf_path_abs == 0.] = 10**1.6
        cos_q_path_abs[cos_q_path_abs == 0.] = 10**1.6

        c1, = ax[i].plot(plot_bins_cos_sf, np.log10(cos_sf_path_abs), c='c', marker='o', ls='--')
        c2, = ax[i].plot(plot_bins_cos_q, np.log10(cos_q_path_abs), c='m', marker='o', ls='--')
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        l1 = ax[i].errorbar(plot_bins_sim_sf, sim_sf_path_abs, yerr=sim_sf_err, c='b', marker='o', ls='--')
        l2 = ax[i].errorbar(plot_bins_sim_q, sim_q_path_abs, yerr=sim_q_err, c='r', marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z),\ $' + plot_lines[i])
       
        ax[i].set_ylim(1.5, 2.9)
        if r200_scaled:
            ax[i].set_xlim(0, 1.1)
        else:
            ax[i].set_xlim(25, 145)

        if i==0:
            ax[i].add_artist(leg1)

    plt.savefig(plot_dir+plot_name)


