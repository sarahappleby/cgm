import numpy as np
from astropy.io import ascii
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pyigm.cgm import cos_halos as pch
import caesar

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0., 0.95)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

caesar_file = '/home/rad/data/m50n512/s50j7k/Groups/m50n512_151.hdf5'
sim = caesar.load(caesar_file, LoadHalo=False)
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

basic_dir = '/home/sapple/cgm/cos_samples/'

cos_halos = pch.COSHalos()
cos_halos_mass = []
cos_halos_ssfr = []
for cos in cos_halos:
        cos = cos.to_dict()
        cos_halos_mass.append(cos['galaxy']['stellar_mass'])
        cos_halos_ssfr.append(cos['galaxy']['ssfr'])
cos_halos_ssfr = np.log10(np.array(cos_halos_ssfr) + 1.e-14)

table_file = '/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/line_table_simple.tex'
table = ascii.read(table_file, format='latex')
cos_dwarfs_mass = table['logM_stellar']
cos_dwarfs_ssfr = table['logsSFR']

for i, item in enumerate(cos_dwarfs_ssfr):
    if '$<$' in item:
        j = item.find('-')
        cos_dwarfs_ssfr[i] = item[j:]
cos_dwarfs_ssfr = np.array(cos_dwarfs_ssfr, dtype=float)

halos_sample_file = basic_dir + 'cos_halos/samples/m50n512_s50j7k_cos_halos_sample.h5'
with h5py.File(halos_sample_file, 'r') as f:
    halos_mass = f['mass'][:]
    halos_ssfr = np.log10(f['ssfr'][:] + 1.e-14)
    halos_ids = np.array(f['gal_ids'][:], dtype=int)
halos_gas_frac = np.log10(gal_gas_frac[halos_ids] + 1.e-3)

dwarfs_sample_file = basic_dir + 'cos_dwarfs/samples/m50n512_s50j7k_cos_dwarfs_sample.h5'
with h5py.File(dwarfs_sample_file, 'r') as f:
    dwarfs_mass = f['mass'][:]
    dwarfs_ssfr = np.log10(f['ssfr'][:] + 1.e-14)
    dwarfs_ids = np.array(f['gal_ids'][:], dtype=int)
dwarfs_gas_frac = np.log10(gal_gas_frac[dwarfs_ids] + 1.e-3)


fig, ax = plt.subplots(1,2, figsize=(14, 6))

ax[0].scatter(cos_halos_mass, cos_halos_ssfr, marker='^', c='gray', s=25, label='COS-Halos')
im = ax[0].scatter(halos_mass, halos_ssfr, c=halos_gas_frac, s=6, cmap=cmap)
cbar = fig.colorbar(im,ax=ax[0], label=r'$\textrm{log} f_{\textrm{gas}}$')
cbar.ax.tick_params(labelsize=12)
ax[0].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[0].set_ylabel(r'$\textrm{log} (sSFR  / \textrm{M}_{\odot}\textrm{yr}^{-1})$')
ax[0].legend(loc=3)


ax[1].scatter(cos_dwarfs_mass, cos_dwarfs_ssfr, marker='^', c='gray', s=25, label='COS-Dwarfs')
im = ax[1].scatter(dwarfs_mass, dwarfs_ssfr, c=dwarfs_gas_frac, s=5, cmap=cmap)
cbar = fig.colorbar(im,ax=ax[1], label=r'$\textrm{log} f_{\textrm{gas}}$')
cbar.ax.tick_params(labelsize=12)
ax[1].set_xlabel(r'$\textrm{log} (M_* / \textrm{M}_{\odot})$')
ax[1].set_ylabel(r'$\textrm{log} (sSFR  / \textrm{M}_{\odot}\textrm{yr}^{-1})$')
ax[1].legend(loc=3)

plt.savefig('cos_surveys_samples.png')
plt.show()
