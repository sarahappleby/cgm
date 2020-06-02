
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from physics import compute_path_length, compute_path_abs

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    cos_survey = sys.argv[1]
    cos_survey = [cos_survey]* 6
    lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    wave_rest = [1215., 2796., 1206., 1548., 1031., 770.409]
    plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
    det_thresh = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]

    model = 'm50n512'
    wind = 's50j7k'
    velocity_width = 300. # km/s
    mlim = np.log10(5.8e8) # lower limit of M*
    plot_dir = 'plots/'

    rho_bins = np.arange(0., 200., 40.)
    plot_bins = rho_bins[:-1] + 20

    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    ax = ax.flatten()

    halo_rho, halo_M, halo_r200, halo_ssfr = get_cos_halos()
    dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr = get_cos_dwarfs()

    halos_rho_long = np.repeat(halo_rho, 20.)
    dwarfs_rho_long = np.repeat(dwarfs_rho, 20.)

    for i, survey in enumerate(cos_survey):

        if survey == 'dwarfs':
            label = 'COS-Dwarfs'
            snap = '151'
            z = 0.
            cos_rho, cos_M, cos_ssfr = dwarfs_rho.copy(), dwarfs_M.copy(), dwarfs_ssfr.copy()
        elif survey == 'halos':
            label = 'COS-Halos'
            snap = '137'
            z = 0.2
            cos_rho, cos_M, cos_ssfr = halo_rho.copy(), halo_M.copy(), halo_ssfr.copy()

        quench = -1.8  + 0.3*z - 9.

        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            cos_M = np.delete(cos_M, 3)
            cos_ssfr = np.delete(cos_ssfr, 3)
            cos_rho = np.delete(cos_rho, 3)

        cos_rho = cos_rho[cos_M > mlim]
        cos_ssfr = cos_ssfr[cos_M > mlim]
        cos_rho_long = np.repeat(cos_rho, 20)

        cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
        with h5py.File(cos_sample_file, 'a') as f:
            keys = list(f.keys())
            if 'path_length_'+lines[i] not in keys:
                vgal = f['vgal_position'][:][:, 2]
                path_length = compute_path_length(vgal, velocity_width, wave_rest[i], z)
                f.create_dataset('path_length_'+lines[i], data=np.array(path_length))

            mass = np.repeat(f['mass'][:], 4)
            ssfr = np.repeat(f['ssfr'][:], 4)
            path_length = np.repeat(f['path_length_'+lines[i]][:], 4)
        ssfr[ssfr < -11.5] = -11.5

        ew_file = 'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_ew_data_lsf.h5'
        with h5py.File(ew_file, 'r') as f:
            ew = f[lines[i]+'_wave_ew'][:]

        # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            ssfr = np.delete(ssfr, np.arange(3*20, 4*20))
            ew = np.delete(ew, np.arange(3*20, 4*20))
            path_length = np.delete(path_length, np.arange(3*20, 4*20))

        sim_sf_path_abs = compute_path_abs(ew[ssfr > quench], cos_rho_long[ssfr > quench], rho_bins, det_thresh[i], path_length[ssfr > quench])
        sim_q_path_abs = compute_path_abs(ew[ssfr < quench], cos_rho_long[ssfr < quench], rho_bins, det_thresh[i], path_length[ssfr < quench])

        if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
            EW, EWerr, EW_less_than = get_cos_dwarfs_civ() #in mA
            EW /= 1000.
            compare = True
        elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
            EW, EWerr = get_cos_dwarfs_lya() # in mA
            EW /= 1000.
            EW = np.delete(EW, 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
            compare = True
        elif (survey == 'halos'):
            EW, EWerr = read_halos_data(lines[i])
            EW = np.abs(EW)
            compare = True
        else:
            compare = False

        if compare:
            EW = EW[cos_M > mlim]
            cos_path_lengths = np.repeat(path_length[0], len(EW))
        
            cos_sf_path_abs = compute_path_abs(EW[cos_ssfr > quench], cos_rho[cos_ssfr > quench], rho_bins, det_thresh[i], cos_path_lengths[cos_ssfr > quench])
            cos_q_path_abs = compute_path_abs(EW[cos_ssfr < quench], cos_rho[cos_ssfr < quench], rho_bins, det_thresh[i], cos_path_lengths[cos_ssfr < quench])
        
            c1, = ax[i].plot(plot_bins, np.log10(cos_sf_path_abs), c='c', marker='o', ls='--')
            c2, = ax[i].plot(plot_bins, np.log10(cos_q_path_abs), c='m', marker='o', ls='--')
            leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)

        l1, = ax[i].plot(plot_bins, np.log10(sim_sf_path_abs), c='b', marker='o', ls='--')
        l2, = ax[i].plot(plot_bins, np.log10(sim_q_path_abs), c='r', marker='o', ls='--')
        if i == 0:
            leg2 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], loc='lower left', fontsize=10.5)

        ax[i].set_xlabel(r'$\rho (\textrm{kpc})$')
        ax[i].set_ylabel(r'$\textrm{log}\ (\textrm{dEW}/ \textrm{d} z)\ $' + plot_lines[i])
        
        if compare:
            if i==0:
                ax[i].add_artist(leg1)

    plt.savefig(plot_dir+model+'_'+wind+'_'+survey+'_rho_path_abs.png')



