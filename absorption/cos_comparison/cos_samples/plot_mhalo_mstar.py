import numpy as np
import h5py
import matplotlib.pyplot as plt
from ignore_gals import get_ignore_cos_mask, get_ignore_simba_gals, make_ignore_mask
from get_cos_info import *
import sys
sys.path.append('/disk01/sapple/tools/')
from colormap import truncate_colormap

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

if __name__ == '__main__':

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.03, 1.0)

    mlim = np.log10(5.8e8)
    ngals_each = 5
    surveys = ['dwarfs', 'halos']
    models = ['m100n1024', 'm50n512', 'm25n256', 'm25n512']
    res_labels = [r'$\textrm{Simba-100/1024}$', r'$\textrm{Simba-50/512}$', r'$\textrm{Simba-25/256}$', r'$\textrm{Simba-25/512}$']
    linestyles = ['-', '-.', '--', ':',]

    _, cos_halos_mstar, cos_halos_r200, _ = get_cos_halos()
    _, cos_dwarfs_mstar, cos_dwarfs_r200, _ = get_cos_dwarfs()
    cos_dwarfs_mhalo = get_cos_dwarfs_mhalo()
    cos_halos_mhalo = get_cos_halos_mhalo()
   
    cos_dwarfs_r200 = cos_dwarfs_r200[cos_dwarfs_mstar > mlim]
    cos_dwarfs_mhalo = cos_dwarfs_mhalo[cos_dwarfs_mstar > mlim]
    cos_dwarfs_mstar = cos_dwarfs_mstar[cos_dwarfs_mstar > mlim] 

    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8.5, 8.5))

    for m, model in enumerate(models):

        axis = ax.flat[m]

        if model in ['m100n1024', 'm25n512', 'm25n256']: wind = 's50'
        elif model in ['m50n512']: wind = 's50j7k'

        ignore_simba_halos_gals, ngals_each = get_ignore_simba_gals(model, 'halos')
        ignore_simba_halos_mask = make_ignore_mask(ngals_each*44, ignore_simba_halos_gals)
        
        ignore_simba_dwarfs_gals, ngals_each = get_ignore_simba_gals(model, 'dwarfs')
        ignore_simba_dwarfs_mask = make_ignore_mask(ngals_each*39, ignore_simba_dwarfs_gals)
        
        ignore_halo_mask = get_ignore_cos_mask(model, 'halos')
        ignore_dwarfs_mask = get_ignore_cos_mask(model, 'dwarfs')

        halos_mhalo_use = cos_halos_mhalo[ignore_halo_mask]
        halos_mstar_use = cos_halos_mstar[ignore_halo_mask]
        dwarfs_mhalo_use = cos_dwarfs_mhalo[ignore_dwarfs_mask]
        dwarfs_mstar_use = cos_dwarfs_mstar[ignore_dwarfs_mask]

        axis.plot(dwarfs_mhalo_use, dwarfs_mstar_use, ls='', marker='^', mec='darkgray', mfc='none', label='COS-Dwarfs')
        axis.plot(halos_mhalo_use, halos_mstar_use, ls='', marker='o', mec='dimgrey', mfc='none', label='COS-Halos')

        mhalo_med = np.zeros(len(dwarfs_mhalo_use) + len(halos_mhalo_use))
        mstar_med = np.zeros(len(dwarfs_mhalo_use) + len(halos_mhalo_use))
        ssfr_med = np.zeros(len(dwarfs_mhalo_use) + len(halos_mhalo_use))

        i = 0
        for s, survey in enumerate(surveys):
            
            if survey == 'dwarfs':
                mask = ignore_simba_dwarfs_mask
                ngals = len(dwarfs_mhalo_use)
            elif survey == 'halos':
                mask = ignore_simba_halos_mask
                ngals = len(halos_mhalo_use)
            sample_dir = f'/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/{model}/cos_{survey}/samples/'
    
            with h5py.File(f'{sample_dir}{model}_{wind}_cos_{survey}_sample.h5', 'r') as cos_sample:
    
                #mhalo[i:i+sum(mask)] = cos_sample['halo_mass'][:][mask]
                #mstar[i:i+sum(mask)] = cos_sample['mass'][:][mask]
                mhalo = cos_sample['halo_mass'][:][mask]
                mstar = cos_sample['mass'][:][mask]
                ssfr = cos_sample['ssfr'][:][mask] + 9

            mhalo_med[i:i+ngals] = np.nanmedian(mhalo.reshape(int(len(mhalo)/ngals_each), ngals_each), axis=1)
            mstar_med[i:i+ngals] = np.nanmedian(mstar.reshape(int(len(mstar)/ngals_each), ngals_each), axis=1)
            ssfr_med[i:i+ngals] = np.nanmedian(ssfr.reshape(int(len(ssfr)/ngals_each), ngals_each), axis=1)

            i += ngals

        im = axis.scatter(mhalo_med, mstar_med, c=ssfr_med, cmap=cmap, marker='.', label=res_labels[m], vmin=-2.5, vmax=0.)

        axis.legend(loc=2)
        axis.set_xlim(11, 14)
        axis.set_ylim(9.5, 12)
        if m in [2, 3]:
            axis.set_xlabel(r'$\textrm{log} (M_{\rm halo} / \textrm{M}_{\odot})$')
        if m in [0, 2]:
            axis.set_ylabel(r'$\textrm{log} (M_{\star} / \textrm{M}_{\odot})$')
  
    #cax = fig.add_axes([0.93, 0.08, 0.04, 0.8])
    #fig.colorbar(im, label=r'$\textrm{log} (sSFR  / \textrm{yr}^{-1})$', ax=cax)
    #cax = fig.add_axes([0.93, 0.55, 0.04, 0.45])
    #fig.colorbar(im, label=r'$\textrm{log} (sSFR  / \textrm{yr}^{-1})$', ax=cax)
    fig.subplots_adjust(wspace=0., hspace=0.)
    fig.colorbar(im, ax=ax, location='right', label=r'$\textrm{log} (sSFR  / \textrm{Gyr}^{-1})$', shrink=0.8)
    #plt.tight_layout()
    plt.savefig('plots/mhalo_mstar_resolution.png')
  
