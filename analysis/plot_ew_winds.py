
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import sys
import numpy as np
from plot_cos_data import plot_dwarfs_civ, plot_dwarfs_lya, plot_halos
from physics import compute_binned_ew


sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def median_cos_groups(ew, num_gals, num_cos):
    new_ew = np.zeros(num_cos)
    ew_low = np.zeros(num_cos)
    ew_high = np.zeros(num_cos)
    for i in range(num_cos):
        data = ew[i*num_gals:(i+1)*num_gals]
        data = np.sort(data)[1:num_gals - 1]
        new_ew[i] = np.nanmedian(data)
        ew_low[i] = np.nanpercentile(data, 25)
        ew_high[i] = np.nanpercentile(data, 75)

    return new_ew, ew_low, ew_high


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

cos_survey = sys.argv[1]

cos_survey = [cos_survey] * 6
lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
plot_lines = [r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                    r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$', r'$\textrm{NeVIII}770$']
det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

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
        
        sf_ew, sf_low, sf_high = compute_binned_ew(ew[ssfr > quench], dist[ssfr > quench], rho_bins)
        q_ew, q_low, q_high = compute_binned_ew(ew[ssfr < quench], dist[ssfr < quench], rho_bins)

        ax[i].plot(plot_bins, np.log10(sf_ew), linestyle=ls[j], c='b', )
        ax[i].plot(plot_bins, np.log10(q_ew), linestyle=ls[j], c='r', )
        if j == 0:
            ax[i].fill_between(plot_bins, np.log10(sf_low), np.log10(sf_high), linestyle=ls[j], color='b', alpha=0.1)
            ax[i].fill_between(plot_bins, np.log10(q_low), np.log10(q_high), linestyle=ls[j], color='r', alpha=0.1)
 

    ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
    ax[i].set_ylim(-2.,ylim)

plt.savefig(plot_dir+model+'_winds_rho_ew.png') 
