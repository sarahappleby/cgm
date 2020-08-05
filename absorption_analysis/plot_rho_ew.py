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

def make_cos_dict(survey, mlim, r200_scaled=False):

    # read in the parameters of the COS galaxies
    cos_dict = {}
    if survey == 'halos':
        cos_dict['rho'], cos_dict['mass'], cos_dict['r200'], cos_dict['ssfr'] = get_cos_halos()
        z = 0.25
    elif survey == 'dwarfs':
        cos_dict['rho'], cos_dict['mass'], cos_dict['r200'], cos_dict['ssfr'] = get_cos_dwarfs()
        z = 0.

    # rescale the impact parameters into kpc/h
    if r200_scaled:
        cos_dict['rho'] = cos_dict['rho'].astype(float)
        cos_dict['rho'] *= h * (1+z) # get in kpc/h

    # remove COS galaxies below the mass threshold
    mass_mask = cos_dict['mass'] > mlim
    for k in cos_dict.keys():
        cos_dict[k] = cos_dict[k][mass_mask]

    return cos_dict


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
    r200_scaled = True
    norients = 8
    ngals_each = 5

    if model == 'm100n1024':
        boxsize = 100000.
    elif model == 'm50n512':
        boxsize = 50000.

    # adjust the filename
    plot_name = model+'_'+wind +'_rho_ew'
    if r200_scaled:
        plot_name += '_scaled'
    plot_name += '.png'

    # read in the parameters of the COS galaxies
    cos_halos_dict = make_cos_dict('halos', mlim, r200_scaled)
    cos_dwarfs_dict = make_cos_dict('dwarfs', mlim, r200_scaled)    

    sim_halos_dict = read_simulation_sample(model, wind, '137', 'halos', norients, lines, r200_scaled)
    sim_halos_dict['rho'] = np.repeat(cos_halos_dict['rho'], norients*ngals_each)

    sim_dwarfs_dict = read_simulation_sample(model, wind, '151', 'dwarfs', norients, lines, r200_scaled)
    sim_dwarfs_dict['rho'] = np.repeat(cos_dwarfs_dict['rho'], norients*ngals_each)
    
    if r200_scaled:
        sim_halos_dict['dist'] = sim_halos_dict['rho'] / sim_halos_dict['r200']
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'] / sim_dwarfs_dict['r200']
        xlabel = r'$\rho / r_{200}$'
    else:
        sim_halos_dict['dist'] = sim_halos_dict['rho'].copy()
        sim_dwarfs_dict['dist'] = sim_dwarfs_dict['rho'].copy()

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    for i, survey in enumerate(cos_survey):

        # assign the COS survey that we want and set some parameters
        if i in [1, 4]:
            cos_dict = cos_dwarfs_dict.copy()
            sim_dict = sim_dwarfs_dict.copy()
            label = 'COS-Dwarfs'
            z = 0.
        else:
            cos_dict = cos_halos_dict.copy()
            sim_dict = sim_halos_dict.copy()
            label = 'COS-Halos'
            z = 0.25
        quench = -1.8  + 0.3*z - 9.

        # removing COS-Dwarfs galaxy 3 for the Lya stuff
        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            for k in cos_dict.keys():
                cos_dict[k] = np.delete(cos_dict[k], 3)
            for k in sim_dict.keys():
                sim_dict[k] = np.delete(sim_dict[k], np.arange(3*norients*ngals_each, 4*norients*ngals_each), axis=0)

        # find the median and range equivalent width for each of the simulation galaxies
        sim_dict['ew_'+lines[i]+'_median'], sim_dict['ew_err_'+lines[i]], sim_dict['median_dist'], sim_dict['ssfr_median'] = \
                median_ew_cos_groups(sim_dict['ew_'+lines[i]], sim_dict['dist'], sim_dict['ssfr'], norients*ngals_each, len(cos_dict['rho'])) 

        # plot the simulation equivalent widths
        mask = sim_dict['ssfr_median'] < quench
        l1 = ax[i].errorbar(sim_dict['median_dist'][mask], sim_dict['ew_'+lines[i]+'_median'][mask], yerr=[sim_dict['ew_err_'+lines[i]][0][mask], sim_dict['ew_err_'+lines[i]][1][mask]], \
                            ms=3.5, marker='s', capsize=4, ls='', c='r')
        mask = sim_dict['ssfr_median'] > quench
        l2 = ax[i].errorbar(sim_dict['median_dist'][mask], sim_dict['ew_'+lines[i]+'_median'][mask], yerr=[sim_dict['ew_err_'+lines[i]][0][mask], sim_dict['ew_err_'+lines[i]][1][mask]], \
                ms=3.5, marker='s', capsize=4, ls='', c='b')
        if i == 0:
            leg1 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], fontsize=10.5, loc=4)

        ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
        ax[i].set_ylim(-2.5, ylim)
        if r200_scaled:
            ax[i].set_xlim(0, 1.25)
        else:
            ax[i].set_xlim(25, 145)

        # plot the COS equivalent widths
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


