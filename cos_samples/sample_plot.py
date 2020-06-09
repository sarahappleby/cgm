import numpy as np
from astropy.io import ascii
import h5py
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
plt.rc('font', family='serif', size=14)

mlim = np.log10(5.8e8)
model = 'm100n1024'

cos_halos_rho, cos_halos_mass, _, cos_halos_ssfr = get_cos_halos()
cos_dwarfs_rho, cos_dwarfs_mass, _, cos_dwarfs_ssfr, cos_dwarfs_less_than = get_cos_dwarfs(return_less_than=True)
cos_dwarfs_ssfr = cos_dwarfs_ssfr[cos_dwarfs_mass > mlim]
cos_dwarfs_less_than = cos_dwarfs_less_than[cos_dwarfs_mass > mlim]
cos_dwarfs_mass = cos_dwarfs_mass[cos_dwarfs_mass > mlim]
cos_halos_ssfr[cos_halos_ssfr < -11.5] = -11.5
cos_dwarfs_ssfr[cos_dwarfs_ssfr < -11.5] = -11.5

basic_dir = '/home/sapple/cgm/cos_samples/'+model+'/'

halos_sample_file = basic_dir + 'cos_halos/samples/'+model+'_s50_cos_halos_sample.h5'
with h5py.File(halos_sample_file, 'r') as f:
    halos_mass = f['mass'][:]
    halos_ssfr = f['ssfr'][:]
    halos_ids = np.array(f['gal_ids'][:], dtype=int)
    halos_gas_frac = np.log10(f['gas_frac'][:] + 1.e-3)

dwarfs_sample_file = basic_dir + 'cos_dwarfs/samples/m100n1024_s50_cos_dwarfs_sample.h5'
with h5py.File(dwarfs_sample_file, 'r') as f:
    dwarfs_mass = f['mass'][:]
    dwarfs_ssfr = f['ssfr'][:]
    dwarfs_ids = np.array(f['gal_ids'][:], dtype=int)
    dwarfs_gas_frac = np.log10(f['gas_frac'][:] + 1.e-3)

halos_ssfr[halos_ssfr < -11.5] = -11.5
dwarfs_ssfr[dwarfs_ssfr < -11.5] = -11.5

# use full cmap for one population, and use clims for full colorbar. then use truncated colormap for the other population
halos_q_bound = np.abs(np.max(halos_gas_frac[halos_ssfr == -11.5])) / (np.abs(np.min(halos_gas_frac)) + np.abs(np.max(halos_gas_frac)))
halos_q_cmap = truncate_colormap(cmap, 0,halos_q_bound)

fig, ax = plt.subplots(figsize=(7, 5))
im = plt.scatter(halos_mass[halos_ssfr > -11.5], halos_ssfr[halos_ssfr > -11.5], c=halos_gas_frac[halos_ssfr > -11.5], s=7, marker='o', cmap=cmap)
plt.colorbar(im, label=r'$\textrm{log} (f_{\textrm{gas}})$')
plt.clim(np.max(halos_gas_frac), np.min(halos_gas_frac))
plt.scatter(halos_mass[halos_ssfr == -11.5], halos_ssfr[halos_ssfr == -11.5], c=halos_gas_frac[halos_ssfr == -11.5], s=35, marker='$\downarrow$', cmap=halos_q_cmap)
plt.scatter(cos_halos_mass, cos_halos_ssfr, marker='x', c='gray', s=25, label='COS-Halos')
plt.xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
plt.ylabel(r'$\textrm{log} (sSFR  / \textrm{M}_{\odot}\textrm{yr}^{-1})$')
plt.ylim(-11.7, -9.)
plt.legend(loc=1)
plt.savefig('plots/cos_halos_sample.png')
plt.clf()


dwarfs_q_bound = np.abs(np.max(dwarfs_gas_frac[dwarfs_ssfr == -11.5])) / (np.abs(np.min(dwarfs_gas_frac)) + np.abs(np.max(dwarfs_gas_frac)))
dwarfs_q_cmap = truncate_colormap(cmap, 0,dwarfs_q_bound)

fig, ax = plt.subplots(figsize=(7, 5))
im = plt.scatter(dwarfs_mass[dwarfs_ssfr > -11.5], dwarfs_ssfr[dwarfs_ssfr > -11.5], c=dwarfs_gas_frac[dwarfs_ssfr > -11.5], s=7, marker='o', cmap=cmap)
plt.colorbar(im, label=r'$f_{\textrm{gas}}$')
plt.clim(np.max(dwarfs_gas_frac), np.min(dwarfs_gas_frac))
plt.scatter(dwarfs_mass[dwarfs_ssfr == -11.5], dwarfs_ssfr[dwarfs_ssfr == -11.5], c=dwarfs_gas_frac[dwarfs_ssfr == -11.5], s=35, marker='$\downarrow$', cmap=dwarfs_q_cmap)
plt.scatter(cos_dwarfs_mass[np.invert(cos_dwarfs_less_than)], cos_dwarfs_ssfr[np.invert(cos_dwarfs_less_than)], marker='x', c='gray', s=25, label='COS-Dwarfs')
plt.scatter(cos_dwarfs_mass[cos_dwarfs_less_than], cos_dwarfs_ssfr[cos_dwarfs_less_than], marker='$\downarrow$', c='gray', s=50)
plt.xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
plt.ylabel(r'$\textrm{log} (sSFR  / \textrm{M}_{\odot}\textrm{yr}^{-1})$')
plt.ylim(-11.7, -9.)
plt.legend(loc=1)
plt.savefig('plots/cos_dwarfs_sample.png')
plt.clf()
