import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from plot_cos_data import *

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                                                                                cmap(np.linspace(minval, maxval, n)))
        return new_cmap

cmap = plt.get_cmap('jet_r')
cmap = truncate_colormap(cmap, 0.05, 1.0)

cos_survey = sys.argv[1]

model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8) # lower limit of M*
lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
halos_ions = ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031',]

cos_sample_file = '/home/sapple/cgm/cos_samples/cos_'+cos_survey+'/samples/'+model+'_'+wind+'_cos_'+cos_survey+'_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    mass = np.repeat(f['mass'][:], 4)
    ssfr = np.repeat(f['ssfr'][:], 4) + 9.

ssfr[ssfr < -2.5] = -2.5 

if cos_survey == 'dwarfs':
    snap = '151'
    from get_cos_info import get_cos_dwarfs
    cos_rho, cos_M, cos_ssfr = get_cos_dwarfs()
    ylim = 0.5
elif cos_survey == 'halos':
    snap = '137'
    from get_cos_info import get_cos_halos
    cos_rho, cos_M, cos_ssfr = get_cos_halos()
    ylim = 1.0

cos_rho = cos_rho[cos_M > mlim]
cos_rho_long = np.repeat(cos_rho, 20.)

ew_file = 'data/cos_'+cos_survey+'_'+model+'_'+snap+'_ew_data.h5'
plot_dir = 'plots/cos_'+cos_survey+'/'

cos_ew = []

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax = ax.flatten()

for i, line in enumerate(lines):
    
    with h5py.File(ew_file, 'r') as f:
        ew = f[line+'_wave_ew'][:]

    if (cos_survey == 'dwarfs') & (line == 'CIV1548'):
        plot_dwarfs_civ(ax[i])
    elif (line == 'H1215') & (cos_survey == 'dwarfs'):
        plot_dwarfs_lya(ax[i])
    elif (cos_survey == 'halos') & (line in halos_ions):
        plot_halos(ax[i], line)

    im = ax[i].scatter(cos_rho_long, np.log10(ew), s=3, c=ssfr, cmap=cmap)
    ax[i].axhline(-1, ls='--', c='k', lw=1)
    ax[i].set_xlabel('Impact parameter')
    ax[i].set_ylabel('EW ' + line)
    ax[i].set_ylim(-2, ylim)
    cbar = fig.colorbar(im,ax=ax[i], label='sSFR')
    ax[i].legend(loc=1)

plt.savefig(plot_dir+'ions_impact_parameter.png')


