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

cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$', 
                r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon

model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8) # lower limit of M*
ylim = 0.7
r200_scaled = True

if model == 'm100n1024':
    boxsize = 100000.
elif model == 'm50n512':
    boxsize = 50000.

plot_dir = 'plots/'

fig, ax = plt.subplots(3, 2, figsize=(12, 14))
ax = ax.flatten()

halo_rho, halo_M, halo_r200, halo_ssfr = get_cos_halos()
dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr = get_cos_dwarfs()


for i, survey in enumerate(cos_survey):

    cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        mass = np.repeat(f['mass'][:], 4)
        ssfr = np.repeat(f['ssfr'][:], 4)
        pos = np.repeat(f['pos'][:], 4)
        r200 = np.repeat(f['halo_r200'][:], 4)
    ssfr[ssfr < -11.5] = -11.5

    if survey == 'dwarfs':
        label = 'COS-Dwarfs'
        snap = '151'
        z = 0.
        cos_rho, cos_M, cos_r200, cos_ssfr = dwarfs_rho, dwarfs_M, dwarfs_r200, dwarfs_ssfr
    elif survey == 'halos':
        label = 'COS-Halos'
        snap = '137'
        z = 0.2 
        cos_rho, cos_M, cos_r200, cos_ssfr = halo_rho, halo_M, halo_r200, halo_ssfr

    cos_rho = cos_rho[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    cos_r200 = cos_r200[cos_M > mlim]
    quench = -1.8  + 0.3*z - 9.

    ew_file = 'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_ew_data_lsf.h5'

    with h5py.File(ew_file, 'r') as f:
        ew = np.log10(f[lines[i]+'_wave_ew'][:])

    ew, ew_err = median_ew_cos_groups(ew, pos, 20, len(cos_rho), boxsize)
    
    # remove index 3 from dwarfs lya because this isnt in the data
    if (survey == 'dwarfs') & (lines[i] == 'H1215'):
        ew = np.delete(ew, 3)
        ew_err = np.delete(ew_err, 3)
        cos_ssfr = np.delete(cos_ssfr, 3)
        cos_rho = np.delete(cos_rho, 3)

    if r200_scaled:
        dist = cos_rho / cos_r200
        xlabel = r'$\rho / r_{200} (\textrm{kpc)$'
    else:
        dist = cos_rho.copy()
        xlabel = r'$\rho (\textrm{kpc})$'

    l1 = ax[i].errorbar(dist[cos_ssfr < quench], ew[cos_ssfr < quench], yerr=ew_err[cos_ssfr < quench], \
                        ms=3.5, marker='s', capsize=4, ls='', c='r')
    l2 = ax[i].errorbar(dist[cos_ssfr > quench], ew[cos_ssfr > quench], yerr=ew_err[cos_ssfr > quench], \
                        ms=3.5, marker='s', capsize=4, ls='', c='b')
    if i == 0:
        leg1 = ax[i].legend([l1, l2], ['Simba SF', 'Simba Q'], fontsize=10.5, loc=4)

    ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylabel(r'$\textrm{log (EW}\  $' + plot_lines[i] + r'$/ \AA  )$')
    ax[i].set_ylim(-2.5, ylim)

    if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
        c1, c2 = plot_dwarfs_civ(ax[i], quench, r200_scaled=r200_scaled)
    elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
        c1, c2 = plot_dwarfs_lya(ax[i], quench, r200_scaled=r200_scaled)
    elif (survey == 'halos'):
        c1, c2 = plot_halos(ax[i], lines[i], quench, r200_scaled)

    leg2 = ax[i].legend([c1, c2], [label+' SF', label+' Q'], loc=3, fontsize=10.5)

    if i == 0:
        ax[i].add_artist(leg1)

plt.savefig(plot_dir+model+'_'+wind+'_rho_ew.png')


