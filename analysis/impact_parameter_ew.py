import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from pyigm.cgm import cos_halos as pch

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                                                                                cmap(np.linspace(minval, maxval, n)))
        return new_cmap

cos_survey = sys.argv[1]
ew_file = sys.argv[2]
plot_dir = sys.argv[3]

model = 'm50n512'
wind = 's50j7k'

cos_sample_file = '/home/sapple/cgm/cos_samples/'+cos_survey+'/samples/'+model+'_'+wind+'_'+cos_survey+'_galaxy_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    mass = np.repeat(f['mass'][:], 4)
    ssfr = np.log10(np.repeat(f['ssfr'][:] *  (10**9.), 4) + 10**(-2.5))

# need to get the impact parameters from the COS-Halos survey data:
cos_halos = pch.COSHalos()
cos_rho = []
for cos in cos_halos:
                cos = cos.to_dict()
                cos_rho.append(cos['rho'])
cos_rho = np.repeat(np.array(cos_rho), 20)

ions = ['H1215', 'MgII2796', 'SiII1260', 'CIV1548', 'OVI1031', 'NeVIII770']

cmap = plt.get_cmap('jet_r')
cmap = truncate_colormap(cmap, 0.03, 1.0)

fig, ax = plt.subplots(3, 2, figsize=(12, 10))
ax = ax.flatten()

for i, ion in enumerate(ions):
    
    with h5py.File(ew_file, 'r') as f:
        ew = f[ion+'_ew'][:]

    im = ax[i].scatter(cos_rho, np.log10(ew), s=3, c=ssfr, cmap=cmap)
    ax[i].axhline(-1, ls='--', c='k', lw=1)
    ax[i].set_xlabel('Impact parameter')
    ax[i].set_ylabel('EW ' + ion)
    ax[i].set_ylim(-2, )
    cbar = fig.colorbar(im,ax=ax[i], label='sSFR')

plt.savefig(plot_dir+'ions_impact_parameter.png')
