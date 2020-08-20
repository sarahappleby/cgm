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
    halos_det_thresh = [0.2, 0.1, 0.1, 0.1]
    dwarfs_det_thresh = [0.2, 0.1]
    norients = 8
    ngals_each = 5
    mlim = np.log10(5.8e8) # lower limit of M*
    r200_scaled = True
    background = 'uvb_hm12'

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
        det_thresh = halos_det_thresh
    elif survey == 'dwarfs':
        z = 0.
        snap = '151'
        nbins_q = 1
        nbins_sf = 3
        lines = dwarfs_lines
        det_thresh = dwarfs_det_thresh

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

    cos_file = '/home/sapple/cgm/absorption_analysis/data/cos_'+survey+'_obs_cfrac_data.h5'
    sim_file = '/home/sapple/cgm/absorption_analysis/data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_'+background+'_sim_cfrac_data.h5'

    if survey == 'halos':
        cos_dict_orig, cos_mmask = make_cos_dict('halos', mlim, r200_scaled)
    elif survey == 'dwarfs':
        cos_dict_orig, cos_mmask = make_cos_dict('dwarfs', mlim, r200_scaled)

    # rescaled the x axis by r200
    if r200_scaled:
        cos_dict_orig['dist'] = cos_dict_orig['rho'] / cos_dict_orig['r200']
    else:
        cos_dict_orig['dist'] = cos_dict_orig['rho'].copy()

    if not os.path.isfile(cos_file):

        # get the bins for the COS data - these nbins ensure there are roughly ~8 galaxies in each bin
        cos_plot_dict = {}
        mask = (cos_dict_orig['ssfr'] > quench)
        cos_plot_dict['dist_bins_sf'], cos_plot_dict['plot_bins_sf'] = do_bins(cos_dict_orig['dist'][mask], nbins_sf)
        cos_plot_dict['dist_bins_q'], cos_plot_dict['plot_bins_q'] = do_bins(cos_dict_orig['dist'][~mask], nbins_q)
        cos_plot_dict['xerr_sf'] = get_xerr_from_bins(cos_plot_dict['dist_bins_sf'], cos_plot_dict['plot_bins_sf'])
        cos_plot_dict['xerr_q'] = get_xerr_from_bins(cos_plot_dict['dist_bins_q'], cos_plot_dict['plot_bins_q'])

        for i, line in enumerate(lines):

            cos_dict = cos_dict_orig.copy()
            mass_mask = cos_mmask.copy()

            if ((model == 'm50n512') & (survey == 'halos')) or ((model == 'm25n512') & (survey == 'dwarfs')):
                mass_mask = np.delete(mass_mask, ignore_cos_gals)
                for k in cos_dict.keys():
                    cos_dict[k] = np.delete(cos_dict[k], ignore_cos_gals)

            # removing COS-Dwarfs galaxy 3 for the Lya stuff
            if (survey == 'dwarfs') & (line == 'H1215'):
                mass_mask = np.delete(mass_mask, 3)
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
            cos_dict['EW'] = cos_dict['EW'][mass_mask]
            cos_dict['EWerr'] = cos_dict['EWerr'][mass_mask]

            if survey == 'halos':
                ew_mask = cos_dict['EW'] > 0.
                for k in cos_dict.keys():
                    cos_dict[k] = cos_dict[k][ew_mask]

            cos_plot_dict['cfrac_'+line+'_sf'], cos_plot_dict['cfrac_'+line+'_poisson_sf'] = \
                    cos_binned_cfrac(cos_dict, (cos_dict['ssfr'] > quench), cos_plot_dict['dist_bins_sf'], det_thresh[i])
            cos_plot_dict['cfrac_'+line+'_q'], cos_plot_dict['cfrac_'+line+'_poisson_q'] = \
                    cos_binned_cfrac(cos_dict, (cos_dict['ssfr'] < quench), cos_plot_dict['dist_bins_q'], det_thresh[i])

        write_dict_to_h5(cos_plot_dict, cos_file)

    else:
        cos_plot_dict = read_dict_from_h5(cos_file)

    if not os.path.isfile(sim_file):

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

        for i, line in enumerate(lines):

            # removing COS-Dwarfs galaxy 3 for the Lya stuff
            if (survey == 'dwarfs') & (line == 'H1215'):
                for k in sim_dict.keys():
                    sim_dict[k] = np.delete(sim_dict[k], np.arange(3*norients*ngals_each, 4*norients*ngals_each), axis=0)

            # get binned medians for the simulation sample
            mask = (sim_dict['ssfr'] > quench)
            sim_plot_dict['ngals_'+line+'_sf'] = get_ngals(sim_dict['dist'][mask], sim_plot_dict['dist_bins_sf'])
            sim_plot_dict['cfrac_'+line+'_sf'], sim_plot_dict['cfrac_'+line+'_cv_std_sf'], sim_plot_dict['cfrac_'+line+'_poisson_sf'] = \
                    sim_binned_cfrac(sim_dict, mask, sim_plot_dict['dist_bins_sf'], det_thresh[i], line, boxsize)
            sim_plot_dict['ngals_'+line+'_q'] = get_ngals(sim_dict['dist'][~mask], sim_plot_dict['dist_bins_q'])
            sim_plot_dict['cfrac_'+line+'_q'], sim_plot_dict['cfrac_'+line+'_cv_std_q'], sim_plot_dict['cfrac_'+line+'_poisson_q'] = \
                    sim_binned_cfrac(sim_dict, ~mask, sim_plot_dict['dist_bins_q'], det_thresh[i], line, boxsize)

        write_dict_to_h5(sim_plot_dict, sim_file)
