# SA: leave this for now as it's not obvious which cos-dwarfs lya measurements correspond to which civ measurements
# check this with the galaxy names in the fits files

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np

sys.path.append('../cos_samples/')
from get_cos_info import *

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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                                                                                cmap(np.linspace(minval, maxval, n)))
        return new_cmap

cmap = plt.get_cmap('jet_r')
cmap = truncate_colormap(cmap, 0.05, 1.0)

cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
det_thresh = np.log10([0.2, 0.2, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8) # lower limit of M*
ylim = 0.7

plot_dir = 'plots/'

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax = ax.flatten()

halo_rho, halo_M, halo_ssfr = get_cos_halos()
dwarfs_rho, dwarfs_M, dwarfs_ssfr = get_cos_dwarfs()

EW_less_than = []

for i, survey in enumerate(cos_survey):

    cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        mass = np.repeat(f['mass'][:], 4)
        ssfr = np.repeat(f['ssfr'][:], 4) + 9.
    ssfr[ssfr < -2.5] = -2.5

    if survey == 'dwarfs':
        snap = '151'
        z = 0.
        cos_rho, cos_M, cos_ssfr = dwarfs_rho, dwarfs_M, dwarfs_ssfr
    elif survey == 'halos':
        snap = '137'
        z = 0.2 
        cos_rho, cos_M, cos_ssfr = halo_rho, halo_M, halo_ssfr

    cos_rho = cos_rho[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    quench = -1.8  + 0.3*z - 9.

    ew_file = 'data/cos_'+survey+'_'+model+'_'+snap+'_ew_data.h5'

    with h5py.File(ew_file, 'r') as f:
        ew = np.log10(f[lines[i]+'_wave_ew'][:])

    ew, ew_low, ew_high = median_cos_groups(ew, 20, len(cos_rho))
    ew_sig_low = np.abs(ew - ew_low)
    ew_sig_high = np.abs(ew_high - ew)
    
    if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
        cos_ew, cos_ewerr, ew_less_than = get_cos_dwarfs_civ()
    elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
        cos_ew, cos_ewerr = get_cos_dwarfs_lya()
    elif (survey == 'halos'):
        cos_ew, cos_ewerr = read_halos_data(lines[i])

    cos_ew = cos_ew[cos_M > mlim]
    cos_ewerr = cos_ewerr[cos_M > mlim]

    dev_ew = cos_ew - ew
    dev_ewerr_hi = np.sqrt((cos_ew**2) * (cos_ewerr**2)  + (ew**2) * (ew_sig_high**2)) 
    dev_ewerr_lo = np.sqrt((cos_ew**2) * (cos_ewerr**2)  + (ew**2) * (ew_sig_low**2))

    ax[i].errorbar(cos_rho, dev_ew, yerr=[dev_ewerr_lo, dev_ewerr_hi], ls='')

    ax[i].legend(loc=1)
    ax[i].set_xlabel('Impact parameter')
    ax[i].set_ylabel('EW ' + lines[i] + '; COS - Simba')


    EW_less_than = []

plt.savefig(plot_dir+'ions_impact_parameter_dev.png')


