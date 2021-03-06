import h5py
import sys
import os
import numpy as np
from analysis_methods import *

sys.path.append('../cos_samples/')
from get_cos_info import make_cos_dict, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    survey = sys.argv[3]

    halos_lines = ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031', ]
    dwarfs_lines = ['H1215', 'CIV1548']
    norients = 8
    ngals_each = 5
    mlim = np.log10(5.8e8) # lower limit of M*
    r200_scaled = True
    background = 'uvb_hm12'
    mass_bins = [10., 10.5, 11.0]
    bin_labels = ['10.0-10.5', '10.5-11.0']

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    if survey == 'halos':
        z = 0.25
        snap = '137'
        nbins_q = 2
        nbins_sf = 3
        lines = halos_lines
    elif survey == 'dwarfs':
        z = 0.
        snap = '151'
        nbins_q = 1
        nbins_sf = 3
        lines = dwarfs_lines

    quench = quench_thresh(z)

    # ignore the galaxies that dont have counterparts in the m50n512 boxes
    if (model == 'm50n512') & (survey == 'halos'):
        ignore_cos_gals = [18, 29]
    if (model == 'm25n512') & (survey == 'dwarfs'):
        ignore_cos_gals = [10, 17, 36]
    if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
        ignore_simba_gals = [list(range(num*norients*ngals_each, (num+1)*norients*ngals_each)) for num in ignore_cos_gals]
        ignore_simba_gals = [item for sublist in ignore_simba_gals for item in sublist]
    else:
        ignore_simba_gals = []

    cos_file = '/home/sapple/cgm/absorption_analysis/data/cos_'+survey+'_obs_ew_med_data.h5'
    sim_file = '/home/sapple/cgm/absorption_analysis/data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_'+background+'_sim_ew_med_data.h5'

    if survey == 'halos':
        cos_dict_orig, cos_mmask = make_cos_dict('halos', mlim, r200_scaled)
    elif survey == 'dwarfs':
        cos_dict_orig, cos_mmask = make_cos_dict('dwarfs', mlim, r200_scaled)

    # rescaled the x axis by r200
    if r200_scaled:
        cos_dict_orig['dist'] = cos_dict_orig['rho'] / cos_dict_orig['r200']
    else:
        cos_dict_orig['dist'] = cos_dict_orig['rho'].copy()

    for m in range(len(mass_bins) - 1):

        with h5py.File(cos_file, 'r') as cf:
            cos_keys = list(cf.keys())
        if not 'EW_'+lines[0]+'_med_sf_'+bin_labels[m] in cos_keys:

            # get the bins for the COS data - these nbins ensure there are roughly ~8 galaxies in each bin
            cos_plot_dict = {}
            cos_mass_mask = (cos_dict_orig['mass'] > mass_bins[m]) & (cos_dict_orig['mass'] < mass_bins[m+1])
            cos_ssfr_mask = (cos_dict_orig['ssfr'] > quench)
            cos_plot_dict['dist_bins_sf_'+bin_labels[m]], cos_plot_dict['plot_bins_sf_'+bin_labels[m]] = \
                    do_bins(cos_dict_orig['dist'][cos_mass_mask * cos_ssfr_mask], nbins_sf)
            cos_plot_dict['dist_bins_q_'+bin_labels[m]], cos_plot_dict['plot_bins_q_'+bin_labels[m]] = \
                    do_bins(cos_dict_orig['dist'][cos_mass_mask * ~cos_ssfr_mask], nbins_q)
            cos_plot_dict['xerr_sf_'+bin_labels[m]] = \
                    get_xerr_from_bins(cos_plot_dict['dist_bins_sf_'+bin_labels[m]], cos_plot_dict['plot_bins_sf_'+bin_labels[m]])
            cos_plot_dict['xerr_q_'+bin_labels[m]] = \
                    get_xerr_from_bins(cos_plot_dict['dist_bins_q_'+bin_labels[m]], cos_plot_dict['plot_bins_q_'+bin_labels[m]])

            for i, line in enumerate(lines):

                cos_dict = cos_dict_orig.copy()
                mass_mask = cos_mass_mask.copy()
                lower_mass_mask = cos_mmask.copy()

                if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
                    mass_mask = np.delete(mass_mask, ignore_cos_gals)
                    lower_mass_mask = np.delete(lower_mass_mask, ignore_cos_gals)
                    for k in cos_dict.keys():
                        cos_dict[k] = np.delete(cos_dict[k], ignore_cos_gals)

                # removing COS-Dwarfs galaxy 3 for the Lya stuff
                if (survey == 'dwarfs') & (line == 'H1215'):
                    mass_mask = np.delete(mass_mask, 3)
                    lower_mass_mask = np.delete(lower_mass_mask, 3)
                    for k in cos_dict.keys():
                        cos_dict[k] = np.delete(cos_dict[k], 3)

                # read in COS observations, set units, remove dwarfs galaxy 3 for Lya, do mass mask
                if (survey == 'dwarfs') & (line == 'CIV1548'):
                    cos_dict['EW'], cos_dict['EWerr'], cos_dict['EW_less_than'] = get_cos_dwarfs_civ() #in mA
                    cos_dict['EW'] /= 1000.
                    cos_dict['EWerr'] /= 1000.
                elif (survey == 'dwarfs') & (line == 'H1215'):
                    cos_dict['EW'], cos_dict['EWerr'] = get_cos_dwarfs_lya() # in mA
                    cos_dict['EW'] /= 1000.
                    cos_dict['EWerr'] /= 1000.
                    cos_dict['EW'] = np.delete(cos_dict['EW'], 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
                    cos_dict['EWerr'] = np.delete(cos_dict['EWerr'], 3)
                elif (survey == 'halos'):
                    cos_dict['EW'], cos_dict['EWerr'] = read_halos_data(line)
                    cos_dict['EW'] = np.abs(cos_dict['EW'])

                if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
                    cos_dict['EW'] = np.delete(cos_dict['EW'], ignore_cos_gals)
                    cos_dict['EWerr'] = np.delete(cos_dict['EWerr'], ignore_cos_gals)
                cos_dict['EW'] = cos_dict['EW'][lower_mass_mask]
                cos_dict['EWerr'] = cos_dict['EWerr'][lower_mass_mask]

                if survey == 'halos':
                    ew_mask = cos_dict['EW'] > 0.
                    mass_mask = mass_mask[ew_mask]
                    for k in cos_dict.keys():
                        cos_dict[k] = cos_dict[k][ew_mask]

                cos_plot_dict['EW_'+lines[i]+'_med_sf_'+bin_labels[m]], cos_plot_dict['EW_'+lines[i]+'_std_sf_'+bin_labels[m]], \
                        cos_plot_dict['EW_'+lines[i]+'_per25_sf_'+bin_labels[m]], cos_plot_dict['EW_'+lines[i]+'_per75_sf_'+bin_labels[m]] = \
                        cos_binned_ew(cos_dict, mass_mask * (cos_dict['ssfr'] > quench), cos_plot_dict['dist_bins_sf_'+bin_labels[m]])
                cos_plot_dict['EW_'+lines[i]+'_med_q_'+bin_labels[m]], cos_plot_dict['EW_'+lines[i]+'_std_q_'+bin_labels[m]], \
                        cos_plot_dict['EW_'+lines[i]+'_per25_q_'+bin_labels[m]], cos_plot_dict['EW_'+lines[i]+'_per75_q_'+bin_labels[m]]  = \
                        cos_binned_ew(cos_dict, mass_mask * (cos_dict['ssfr'] < quench), cos_plot_dict['dist_bins_q_'+bin_labels[m]])

            write_dict_to_h5(cos_plot_dict, cos_file)
    
        else:
            cos_plot_dict = read_dict_from_h5(cos_file)

        with h5py.File(sim_file, 'r') as sf:
            sim_keys = list(sf.keys())
        if not 'EW_'+lines[0]+'_med_sf_'+bin_labels[m] in sim_keys:

            # create the dicts to hold the simulation sample data
            sim_dict = read_simulation_sample(model, wind, snap, survey, background, norients, lines, r200_scaled)
            sim_dict['rho'] = np.repeat(cos_dict_orig['rho'], norients*ngals_each)
            

            # rescaled the x axis by r200
            if r200_scaled:
                sim_dict['dist'] = sim_dict['rho'] / sim_dict['r200']
            else:
                sim_dict['dist'] = sim_dict['rho'].copy()

            if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
                for k in sim_dict.keys():
                    sim_dict[k] = np.delete(sim_dict[k], ignore_simba_gals, axis=0)

            sim_plot_dict = get_equal_bins(model, survey, r200_scaled)
            for k in list(sim_plot_dict.keys()):
                sim_plot_dict[k+'_'+bin_labels[m]] = sim_plot_dict[k].copy()
                del sim_plot_dict[k]

            for i, line in enumerate(lines):

                # removing COS-Dwarfs galaxy 3 for the Lya stuff
                if (survey == 'dwarfs') & (line == 'H1215'):
                    for k in sim_dict.keys():
                        sim_dict[k] = np.delete(sim_dict[k], np.arange(3*norients*ngals_each, 4*norients*ngals_each), axis=0)

                # get binned medians for the simulation sample
                mass_mask = (sim_dict['mass'] > mass_bins[m]) & (sim_dict['mass'] < mass_bins[m+1])
                ssfr_mask = (sim_dict['ssfr'] > quench)
                mask = mass_mask * ssfr_mask
                sim_plot_dict['ngals_'+line+'_sf_'+bin_labels[m]] = get_ngals(sim_dict['dist'][mask], sim_plot_dict['dist_bins_sf_'+bin_labels[m]])
                sim_plot_dict['EW_'+line+'_med_sf_'+bin_labels[m]], sim_plot_dict['EW_'+line+'_cosmic_std_sf_'+bin_labels[m]] = \
                        sim_binned_ew(sim_dict, mask, sim_plot_dict['dist_bins_sf_'+bin_labels[m]], line, boxsize)
                sim_plot_dict['ngals_'+line+'_q_'+bin_labels[m]] = get_ngals(sim_dict['dist'][~mask], sim_plot_dict['dist_bins_q_'+bin_labels[m]])
                sim_plot_dict['EW_'+line+'_med_q_'+bin_labels[m]], sim_plot_dict['EW_'+line+'_cosmic_std_q_'+bin_labels[m]] = \
                        sim_binned_ew(sim_dict, ~mask, sim_plot_dict['dist_bins_q_'+bin_labels[m]], line, boxsize)

            write_dict_to_h5(sim_plot_dict, sim_file)
