
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from plot_cos_data import plot_dwarfs_civ, plot_dwarfs_lya, plot_halos
from physics import compute_cfrac


sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs, get_cos_dwarfs_lya, get_cos_dwarfs_civ


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

cos_survey = sys.argv[1]

cos_survey = [cos_survey] * 6
lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
det_thresh = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

model = 'm50n512'
winds = ['s50j7k', 's50nojet', 's50nox', 's50noagn']
ls = ['-', '--', '-.', ':']

mlim = np.log10(5.8e8) # lower limit of M*
ylim = 0.5
r200_scaled = False

if not r200_scaled:
    rho_bins = np.arange(0., 200., 40.)
    plot_bins = rho_bins[:-1] + 20

plot_dir = 'plots/'

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

    if survey == 'dwarfs':
        label = 'COS-Dwarfs'
        snap = '151'
        z = 0.
        cos_rho, cos_M, cos_ssfr = dwarfs_rho, dwarfs_M, dwarfs_ssfr
    elif survey == 'halos':
        label = 'COS-Halos'
        snap = '137'
        z = 0.2
        cos_rho, cos_M, cos_ssfr = halo_rho, halo_M, halo_ssfr

    if (survey == 'dwarfs') & (lines[i] == 'H1215'):
        cos_M = np.delete(cos_M, 3)
        cos_ssfr = np.delete(cos_ssfr, 3)
        cos_rho = np.delete(cos_rho, 3)

    cos_rho = cos_rho[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    cos_rho_long = np.repeat(cos_rho, 20)
    quench = -1.8  + 0.3*z - 9.

    if r200_scaled:
        dist = cos_rho_long / r200
        xlabel = r'$\rho / r_{200} (\textrm{kpc)$'
    else:
        dist = cos_rho_long.copy()
        xlabel = r'$\rho (\textrm{kpc})$'

    if i == 1:
        ax[i].add_artist(leg)

    for j, w in enumerate(winds):

        cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+w+'_cos_'+survey+'_sample.h5'
        with h5py.File(cos_sample_file, 'r') as f:
            mass = np.repeat(f['mass'][:], 4)
            ssfr = np.repeat(f['ssfr'][:], 4)
        ssfr[ssfr < -11.5] = -11.5

        ew_file = 'data/cos_'+survey+'_'+model+'_'+w+'_'+snap+'_ew_data_lsf.h5'
        with h5py.File(ew_file, 'r') as f:
            ew = f[lines[i]+'_wave_ew'][:]

        if (survey == 'dwarfs') & (lines[i] == 'H1215'):
            ssfr = np.delete(ssfr, np.arange(3*20, 4*20))
            ew = np.delete(ew, np.arange(3*20, 4*20))
       
        sim_sf_cfrac = compute_cfrac(ew[ssfr > quench], cos_rho_long[ssfr > quench], rho_bins, det_thresh[i])
        sim_q_cfrac = compute_cfrac(ew[ssfr < quench], cos_rho_long[ssfr < quench], rho_bins, det_thresh[i])

        ax[i].plot(plot_bins, sim_sf_cfrac, linestyle=ls[j], c='b', )
        ax[i].plot(plot_bins, sim_q_cfrac, linestyle=ls[j], c='r', )
 
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

        cos_sf_cfrac = compute_cfrac(EW[cos_ssfr > quench], cos_rho[cos_ssfr > quench], rho_bins, det_thresh[i])
        cos_q_cfrac = compute_cfrac(EW[cos_ssfr < quench], cos_rho[cos_ssfr < quench], rho_bins, det_thresh[i])

        c1, = ax[i].plot(plot_bins, cos_sf_cfrac, c='c', marker='o', ls='--', label=label+' SF')
        c2, = ax[i].plot(plot_bins, cos_q_cfrac, c='m', marker='o', ls='--', label=label+' Q')
        leg1 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], fontsize=10.5, loc=1)
        if i == 1:
            ax[i].add_artist(leg)

    ax[i].set_xlabel(xlabel)
    ax[i].set_ylabel(r'$\textrm{Covering fraction},\ $' + plot_lines[i])
    ax[i].set_ylim(0, 1.1)


plt.savefig(plot_dir+model+'_winds_rho_cfrac.png') 
