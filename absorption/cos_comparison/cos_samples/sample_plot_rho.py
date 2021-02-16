# Plot the galaxy sample with rho/r200 as the colorbar

import numpy as np
from astropy.io import ascii
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pyigm.cgm import cos_halos as pch

from get_cos_info import *

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_markers_errorbars(ssfr):

    quench_ids = np.arange(len(ssfr), dtype=int)[ssfr < -11.5]
    n_quench = len(quench_mask)

    quench_errors = np.zeros(len(ssfr))
    quench_errors[quench_ids] = 0.1
    quench_markers = ['o'] * len(ssfr)
    none_markers = [None] * n_quench
    from operator import setitem
    for i, n in zip(quench_ids, none_markers):
        setitem(quench_markers, i, n)

    return quench_markers, quench_errors


cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0., 0.95)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

mlim = np.log10(5.8e8)
ngals_each = 5
model = sys.argv[1]
wind = sys.argv[2]

cos_halos_rho, cos_halos_mass, cos_halos_r200, cos_halos_ssfr = get_cos_halos()
cos_dwarfs_rho, cos_dwarfs_mass, cos_dwarfs_r200, cos_dwarfs_ssfr, cos_dwarfs_less_than = get_cos_dwarfs(return_less_than=True)
cos_dwarfs_ssfr = cos_dwarfs_ssfr[cos_dwarfs_mass > mlim]
cos_dwarfs_rho = cos_dwarfs_rho[cos_dwarfs_mass > mlim]
cos_dwarfs_r200 = cos_dwarfs_r200[cos_dwarfs_mass > mlim]
cos_dwarfs_less_than = cos_dwarfs_less_than[cos_dwarfs_mass > mlim]
cos_dwarfs_mass = cos_dwarfs_mass[cos_dwarfs_mass > mlim]
cos_halos_ssfr[cos_halos_ssfr < -11.5] = -11.5
cos_dwarfs_ssfr[cos_dwarfs_ssfr < -11.5] = -11.5
cos_halos_dist = cos_halos_rho / cos_halos_r200
cos_dwarfs_dist = cos_dwarfs_rho / cos_dwarfs_r200

basic_dir = '/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/'+model+'/'

halos_sample_file = basic_dir + 'cos_halos/samples/'+model+'_'+wind+'_cos_halos_sample.h5'
with h5py.File(halos_sample_file, 'r') as f:
    halos_mass = f['mass'][:]
    halos_ssfr = f['ssfr'][:]
    halos_r200 = f['halo_r200'][:]
    halos_ids = np.array(f['gal_ids'][:], dtype=int)
    halos_gas_frac = np.log10(f['gas_frac'][:] + 1.e-3)
halos_rho = np.repeat(cos_halos_rho, ngals_each)
halos_dist = halos_rho / halos_r200

dwarfs_sample_file = basic_dir + 'cos_dwarfs/samples/'+model+'_'+wind+'_cos_dwarfs_sample.h5'
with h5py.File(dwarfs_sample_file, 'r') as f:
    dwarfs_mass = f['mass'][:]
    dwarfs_ssfr = f['ssfr'][:]
    dwarfs_r200 = f['halo_r200'][:]
    dwarfs_ids = np.array(f['gal_ids'][:], dtype=int)
    dwarfs_gas_frac = np.log10(f['gas_frac'][:] + 1.e-3)
dwarfs_rho = np.repeat(cos_dwarfs_rho, ngals_each)
dwarfs_dist = dwarfs_rho / dwarfs_r200

halos_ssfr[halos_ssfr < -11.5] = -11.5
dwarfs_ssfr[dwarfs_ssfr < -11.5] = -11.5

all_ssfr = np.concatenate((dwarfs_ssfr, halos_ssfr))
all_mass = np.concatenate((dwarfs_mass, halos_mass))
all_dist = np.concatenate((dwarfs_dist, halos_dist))

fig, ax = plt.subplots(figsize=(7, 5))

im = plt.scatter(all_mass[all_ssfr > -11.5], all_ssfr[all_ssfr > -11.5], c=all_dist[all_ssfr > -11.5], s=5, marker='o', cmap=cmap)
plt.colorbar(im, label=r'$\rho / r_{200}$')
plt.clim(0., 1.5)
plt.scatter(all_mass[all_ssfr == -11.5], all_ssfr[all_ssfr == -11.5], c=all_dist[all_ssfr == -11.5], s=45, marker='$\downarrow$', cmap=cmap)

plt.scatter(cos_halos_mass[cos_halos_ssfr > -11.5], cos_halos_ssfr[cos_halos_ssfr > -11.5],
            marker='o', edgecolors='dimgrey', facecolors='none', s=50, label='COS-Halos')
plt.scatter(cos_halos_mass[cos_halos_ssfr == -11.5], cos_halos_ssfr[cos_halos_ssfr == -11.5],
            marker='$\downarrow$', c='dimgrey', s=55)
plt.scatter(cos_dwarfs_mass[np.invert(cos_dwarfs_less_than)], cos_dwarfs_ssfr[np.invert(cos_dwarfs_less_than)],
            marker='^', edgecolors='darkgray', facecolors='none', s=50, label='COS-Dwarfs')
plt.scatter(cos_dwarfs_mass[cos_dwarfs_less_than], cos_dwarfs_ssfr[cos_dwarfs_less_than],
            marker='$\downarrow$', c='darkgray', s=55)

plt.xlabel(r'$\textrm{log} (M_{\star} / \textrm{M}_{\odot})$')
plt.ylabel(r'$\textrm{log} (sSFR  / \textrm{yr}^{-1})$')
plt.ylim(-11.7, -8.8)
plt.legend(loc=1)
plt.savefig('/home/sapple/cgm/cos_samples/plots/'+model+'_'+wind+'_cos_sample_rho.png', bbox_inches = 'tight')
plt.clf()
