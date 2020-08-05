import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from physics import do_bins, do_exclude_outliers, sim_binned_ew, cos_binned_ew

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

def read_simulation_sample(model, wind, snap, survey, norients, lines, r200_scaled):

    data_dict = {}
    cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        data_dict['mass'] = np.repeat(f['mass'][:], norients)
        data_dict['ssfr'] = np.repeat(f['ssfr'][:], norients)
        data_dict['pos'] = np.repeat(f['position'][:], norients, axis=0)
        data_dict['r200'] = np.repeat(f['halo_r200'][:], norients)
    data_dict['ssfr'][data_dict['ssfr'] < -11.5] = -11.5

    for line in lines:
        # Read in the equivalent widths of the simulation galaxies spectra
        ew_file = 'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_ew_data_lsf.h5'
        with h5py.File(ew_file, 'r') as f:
            data_dict['ew_'+line] = f[line+'_wave_ew'][:]

    return data_dict

if __name__ == '__main__':

    # set some parameters
    cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
    lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
    det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

    model = 'm100n1024'
    wind = 's50'
    mlim = np.log10(5.8e8) # lower limit of M*
    plot_dir = 'plots/'
    r200_scaled = True
    do_equal_bins = False # same bin spacing for all subsets
    h = 0.68
    nbins_sim = 4
    nbins_cos = 3
    out = 5.
    ylim = 0.7

    # set plot name according to parameters
    plot_name = model+'_'+wind +'_rho_ew_med'
    if r200_scaled:
        plot_name += '_scaled'
    if do_equal_bins:
        plot_name += '_equal_bins'
    if plot_name[-1] == '_': plot_name = plot_name[:-1]
    plot_name += '.png'

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    cos_halos_dict = make_cos_dict('halos', mlim, r200_scaled)
    cos_dwarfs_dict = make_cos_dict('dwarfs', mlim, r200_scaled)

    sim_halos_dict = read_simulation_sample(model, wind, '137', 'halos', norients, lines, r200_scaled)
    sim_halos_dict['rho'] = np.repeat(cos_halos_dict['rho'], norients*ngals_each)

    sim_dwarfs_dict = read_simulation_sample(model, wind, '151', 'dwarfs', norients, lines, r200_scaled)
    sim_dwarfs_dict['rho'] = np.repeat(cos_dwarfs_dict['rho'], norients*ngals_each)

    if r200_scaled:
        sim_halos_dict['dist'] = sim_halos_dict['rho'] / sim_halos_dict['r200']
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'] / sim_dwarfs_dict['r200']
        cos_halos_dict['dist'] = cos_halos_dict['rho'] / cos_halos_dict['r200']
        cos_dwarfs_dict['dist'] = cos_dwarfs_dict['rho'] / cos_dwarfs_dict['r200']
        xlabel = r'$\rho / r_{200}$'
    else:
        sim_halos_dict['dist'] = sim_halos_dict['rho'].copy()
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'].copy()
        cos_halos_dict['dist'] = cos_halos_dict['rho'].copy()
        cos_dwarfs_dict['dist'] = cos_dwarfs_dict['rho'].copy()
        xlabel = r'$\rho (\textrm{kpc})$'

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
    else:

        # fix this for both dwarfs and halos

        _sim_halos_dict = do_exclude_outliers(sim_halos_dict, out)
        _sim_dwarfs_dict = do_exclude_outliers(sim_dwarfs_dict, out)
        mask = (sim_dict['ssfr'] > quench)
            rho_bins_sim_sf, plot_bins_sim_sf = do_bins(sim_dict['dist'][mask], nbins_sim)
            mask = (sim_dict['ssfr'] < quench)
            rho_bins_sim_q, plot_bins_sim_q = do_bins(sim_dict['dist'][mask], nbins_sim)
            mask = (cos_dict['ssfr'] > quench)
            rho_bins_cos_sf, plot_bins_cos_sf = do_bins(cos_dict['dist'][mask], nbins_cos)
            mask = (cos_dict['ssfr'] < quench)
            rho_bins_cos_q, plot_bins_cos_q = do_bins(cos_dict['dist'][mask], nbins_cos)


    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        if survey == 'dwarfs':
            cos_dict = cos_dwarfs_dict.copy()
            sim_dict = sim_dwarfs_dict.copy()
            label = 'COS-Dwarfs'
            z = 0.
        elif survey == 'halos':
            cos_dict = cos_halos_dict.copy()
            sim_dict = sim_halos_dict.copy()
            label = 'COS-Halos'
            z = 0.25
        quench = -1.8  + 0.3*z - 9.

        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            mass_mask = np.delete(mass_mask, 3)
            for k in cos_dict.keys():
                cos_dict[k] = np.delete(cos_dict[k], 3)
            for k in sim_dict.keys():
                sim_dict[k] = np.delete(sim_dict[k], np.arange(3*20, 4*20), axis=0)

        if not do_equal_bins:
            sim_dict = do_exclude_outliers(sim_dict, out)
            mask = (sim_dict['ssfr'] > quench)
            rho_bins_sim_sf, plot_bins_sim_sf = do_bins(sim_dict['dist'][mask], nbins_sim)
            mask = (sim_dict['ssfr'] < quench)
            rho_bins_sim_q, plot_bins_sim_q = do_bins(sim_dict['dist'][mask], nbins_sim)
            mask = (cos_dict['ssfr'] > quench)
            rho_bins_cos_sf, plot_bins_cos_sf = do_bins(cos_dict['dist'][mask], nbins_cos)
            mask = (cos_dict['ssfr'] < quench)
            rho_bins_cos_q, plot_bins_cos_q = do_bins(cos_dict['dist'][mask], nbins_cos)

        mask = sim_dict['ssfr'] > quench
        sim_sf_ew, sim_sf_err = sim_binned_ew(sim_dict, mask, rho_bins_sim_sf, boxsize)
        mask = sim_dict['ssfr'] < quench
        sim_q_ew, sim_q_err = sim_binned_ew(sim_dict, mask, rho_bins_sim_q, boxsize)

        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            cos_dict['EW'], cos_dict['EWerr'], cos_dict['EW_less_than'] = get_cos_dwarfs_civ() #in mA
            cos_dict['EW'] /= 1000.
            cos_dict['EWerr'] /= 1000.
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            cos_dict['EW'], cos_dict['EWerr'] = get_cos_dwarfs_lya() # in mA
            cos_dict['EW'] /= 1000.
            cos_dict['EWerr'] /= 1000.
            cos_dict['EW'] = np.delete(cos_dict['EW'], 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
            cos_dict['EWerr'] = np.delete(cos_dict['EWerr'], 3)
        elif (survey == 'halos'):
            cos_dict['EW'], cos_dict['EWerr'] = read_halos_data(lines[i])
            cos_dict['EW'] = np.abs(cos_dict['EW'])
        cos_dict['EW'] = cos_dict['EW'][mass_mask]
        cos_dict['EWerr'] = cos_dict['EWerr'][mass_mask]

        if survey == 'halos':
            ew_mask = cos_dict['EW'] > 0.
            for k in cos_dict.keys():
                cos_dict[k] = cos_dict[k][ew_mask]

        #cos_sf_ew, cos_sf_err = cos_binned_ew(cos_dict, (cos_dict['ssfr'] > quench), rho_bins_cos_sf)
        #cos_q_ew, cos_q_err = cos_binned_ew(cos_dict, (cos_dict['ssfr'] < quench), rho_bins_cos_q)

        #c1 = ax[i].errorbar(plot_bins_cos_sf, cos_sf_ew, yerr=cos_sf_err, capsize=4, c='c', marker='o', ls='--', label=label+' SF')
        #c2 = ax[i].errorbar(plot_bins_cos_q, cos_q_ew, yerr=cos_q_err, capsize=4, c='m', marker='o', ls='--', label=label+' Q')
        #c1, = ax[i].plot(plot_bins_cos_sf, cos_sf_ew, c='c', marker='o', ls='--', label=label+' SF')
        #c2, = ax[i].plot(plot_bins_cos_q, cos_q_ew, c='m', marker='o', ls='--', label=label+' Q')
        c1 = ax[i].errorbar(cos_dict['cos_dist'][cos_dict['ssfr'] > quench], cos_dict['EW'][cos_dict['ssfr'] > quench], 
                            yerr=cos_dict['EWerr'][cos_dict['ssfr'] > quench], capsize=4, c='c', marker='o', label=label+' SF')
        c2 = ax[i].errorbar(cos_dict['cos_dist'][cos_dict['ssfr'] < quench], cos_dict['EW'][cos_dict['ssfr'] < quench], 
                            yerr=cos_dict['EWerr'][cos_dict['ssfr'] < quench], capsize=4, c='m', marker='o', label=label+' Q')
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        l1 = ax[i].errorbar(plot_bins_sim_sf, sim_sf_ew, yerr=sim_sf_err, capsize=4, c='b', marker='o', ls='--')
        l2 = ax[i].errorbar(plot_bins_sim_q, sim_q_ew, yerr=sim_q_err, capsize=4, c='r', marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

        ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
        ax[i].set_ylim(-2, ylim)
        if r200_scaled:
            ax[i].set_xlim(0, 1.1)
        else:
            ax[i].set_xlim(25, 145)

    plt.savefig(plot_dir+plot_name)
