import h5py
import sys
import os
import numpy as np
from analysis_methods import *

sys.path.append('../cos_samples/')
from get_cos_info import make_cos_dict, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ
from ignore_gals import *

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    survey = sys.argv[3]

    sim_lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    cos_dwarfs_lines = ['H1215', 'CIV1548']
    cos_halos_lines = ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031']
    sim_det_thresh = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    cos_halos_det_thresh = [0.2, 0.1, 0.1, 0.1]
    cos_dwarfs_det_thresh = [0.2, 0.1]
    norients = 8
    ngals_each = 5
    mlim = np.log10(5.8e8) # lower limit of M*
    r200_scaled = True
    background = 'uvb_fg20'

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.
    elif model == 'm25n512':
        boxsize = 25000.
    elif model == 'm25n256':
        boxsize = 25000.

    if survey == 'halos':
        z = 0.25
        snap = '137'
        cos_lines = cos_halos_lines
        cos_det_thresh = cos_halos_det_thresh
        mass_bins = [10., 10.5, 11.0]
        mass_bin_labels = ['10.0-10.5', '10.5-11.0', '>11.0']
    elif survey == 'dwarfs':
        z = 0.
        snap = '151'
        cos_lines = cos_dwarfs_lines
        cos_det_thresh = cos_dwarfs_det_thresh
        mass_bins = [9.0, 9.5, 10., 10.5]
        mass_bin_labels = ['9.0-9.5', '9.5-10.0', '10.0-10.5']

    quench = quench_thresh(z)
    
    # ignore the galaxies that dont have counterparts in Simba
    ignore_simba_gals, ngals_each = get_ignore_simba_gals(model, survey)
    ignore_cos_gals, ngals_each = get_ignore_cos_gals(model, survey)
    ignore_los = get_ignore_los(ignore_simba_gals)

    if survey == 'halos':
        cos_dict_orig, cos_mmask = make_cos_dict('halos', mlim, r200_scaled)
    elif survey == 'dwarfs':
        cos_dict_orig, cos_mmask = make_cos_dict('dwarfs', mlim, r200_scaled)

    basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/absorption_analysis/'
    # rescaled the x axis by r200
    if r200_scaled:
        cos_dict_orig['dist'] = cos_dict_orig['rho'] / cos_dict_orig['r200']
        cos_file = basic_dir+'data/cos_'+survey+'_obs_cfrac_data_mass_scaled.h5'
        sim_file = basic_dir+'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_'+background+'_sim_cfrac_data_mass_scaled.h5'
    else:
        cos_dict_orig['dist'] = cos_dict_orig['rho'].copy()
        cos_file = basic_dir+'data/cos_'+survey+'_obs_cfrac_mass_data.h5'
        sim_file = basic_dir+'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_'+background+'_sim_cfrac_mass_data.h5'

    if not os.path.isfile(cos_file):

        # get the bins for the COS data - these nbins ensure there are roughly ~8 galaxies in each bin
        cos_plot_dict = get_equal_bins_mass(survey, data='cos', r200_scaled=r200_scaled)
        cos_plot_dict['xerr'] = get_xerr_from_bins(cos_plot_dict['dist_bins'], cos_plot_dict['plot_bins'])

        for i, line in enumerate(cos_lines):

            cos_dict = cos_dict_orig.copy()
            mass_mask = cos_mmask.copy()

            # removing COS-Dwarfs galaxy 3 for the Lya stuff
            if (survey == 'dwarfs') & (line == 'H1215'):
                mass_mask = np.delete(mass_mask, 3)
                for k in cos_dict.keys():
                    cos_dict[k] = np.delete(cos_dict[k], 3)

            for k in cos_dict.keys():
                cos_dict[k] = np.delete(cos_dict[k], ignore_cos_gals)

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

            cos_dict['EW'] = cos_dict['EW'][mass_mask]
            cos_dict['EWerr'] = cos_dict['EWerr'][mass_mask]
            cos_dict['EW'] = np.delete(cos_dict['EW'], ignore_cos_gals)
            cos_dict['EWerr'] = np.delete(cos_dict['EWerr'], ignore_cos_gals)

            if survey == 'halos':
                ew_mask = cos_dict['EW'] > 0.
                for k in cos_dict.keys():
                    cos_dict[k] = cos_dict[k][ew_mask]
            
            for m, mass_label in enumerate(mass_bin_labels):

                if m != len(mass_bins) -1:
                    mass_mask = (cos_dict['mass'] > mass_bins[m]) & (cos_dict['mass'] < mass_bins[m+1])
                else:
                    mass_mask = cos_dict['mass'] > mass_bins[m]

                cos_plot_dict[f'cfrac_{line}_{mass_label}'], cos_plot_dict[f'cfrac_{line}_poisson_{mass_label}'] = \
                        cos_binned_cfrac(cos_dict, mass_mask, cos_plot_dict['dist_bins'], cos_det_thresh[i])

        write_dict_to_h5(cos_plot_dict, cos_file)

    if not os.path.isfile(sim_file):

        # create the dicts to hold the simulation sample data
        sim_dict_orig = read_simulation_sample(model, wind, snap, survey, background, norients, sim_lines, r200_scaled)
        sim_dict_orig['rho'] = np.repeat(cos_dict_orig['rho'], norients*ngals_each)

        # rescaled the x axis by r200
        if r200_scaled:
            sim_dict_orig['dist'] = sim_dict_orig['rho'] / sim_dict_orig['r200']
        else:
            sim_dict_orig['dist'] = sim_dict_orig['rho'].copy()

        for k in sim_dict_orig.keys():
            sim_dict_orig[k] = np.delete(sim_dict_orig[k], ignore_los, axis=0)

        sim_plot_dict = get_equal_bins_mass(survey, r200_scaled=r200_scaled)

        for i, line in enumerate(sim_lines):

            sim_dict = sim_dict_orig.copy()

            # removing COS-Dwarfs galaxy 3 for the Lya stuff
            if (survey == 'dwarfs') & (line == 'H1215'):
                for k in sim_dict.keys():
                    sim_dict[k] = np.delete(sim_dict[k], np.arange(3*norients*ngals_each, 4*norients*ngals_each), axis=0)

            for m, mass_label in enumerate(mass_bin_labels):
                if m != len(mass_bins) -1:
                    mass_mask = (sim_dict['mass'] > mass_bins[m]) & (sim_dict['mass'] < mass_bins[m+1])
                else:
                    mass_mask = sim_dict['mass'] > mass_bins[m]

                # get binned medians for the simulation sample
                sim_plot_dict[f'ngals_{line}_{mass_label}'] = get_ngals(sim_dict['dist'][mass_mask], sim_plot_dict['dist_bins'])
                sim_plot_dict[f'cfrac_{line}_{mass_label}'], sim_plot_dict[f'cfrac_{line}_cv_std_{mass_label}'], sim_plot_dict[f'cfrac_{line}_poisson_{mass_label}'] = \
                        sim_binned_cfrac(sim_dict, mass_mask, sim_plot_dict['dist_bins'], sim_det_thresh[i], line, boxsize)

        write_dict_to_h5(sim_plot_dict, sim_file)
