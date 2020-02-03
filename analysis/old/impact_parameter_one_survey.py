import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np
from plot_cos_data import plot_dwarfs_civ, plot_dwarfs_lya, plot_halos

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

cos_survey = sys.argv[1]

model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8) # lower limit of M*
lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
halos_ions = ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031',]
det_thresh = np.log10([0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) # check CIV with Rongmon, check NeVIII with Jessica?

cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+cos_survey+'/samples/'+model+'_'+wind+'_cos_'+cos_survey+'_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    mass = np.repeat(f['mass'][:], 4)
    ssfr = np.repeat(f['ssfr'][:], 4) + 9.

ssfr[ssfr < -2.5] = -2.5

if cos_survey == 'dwarfs':
    snap = '151'
    z = 0.
    from get_cos_info import get_cos_dwarfs
    cos_rho, cos_M, cos_ssfr = get_cos_dwarfs()
    ylim = 0.5
elif cos_survey == 'halos':
    snap = '137'
    z = 0.2
    
    from get_cos_info import get_cos_halos
    cos_rho, cos_M, cos_ssfr = get_cos_halos()
    ylim = 1.0

cos_rho = cos_rho[cos_M > mlim]
cos_ssfr = cos_ssfr[cos_M > mlim]
quench = -1.8  + 0.3*z - 9.

ew_file = 'data/cos_'+cos_survey+'_'+model+'_'+snap+'_ew_data.h5'
plot_dir = 'plots/cos_'+cos_survey+'/'

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
ax = ax.flatten()

for i, line in enumerate(lines):
    

    print ('\n '+line+'\n')
    
    with h5py.File(ew_file, 'r') as f:
        ew = np.log10(f[line+'_wave_ew'][:])

    ew, ew_low, ew_high = median_cos_groups(ew, 20, len(cos_rho))
    ew_sig_low = np.abs(ew - ew_low)
    ew_sig_high = np.abs(ew_high - ew)

    ax[i].errorbar(cos_rho[cos_ssfr < quench], ew[cos_ssfr < quench], yerr=[ew_sig_low[cos_ssfr < quench], ew_sig_high[cos_ssfr < quench]], 
                    ms=3.5, marker='s', capsize=4, ls='', c='r')
    ax[i].errorbar(cos_rho[cos_ssfr > quench], ew[cos_ssfr > quench], yerr=[ew_sig_low[cos_ssfr > quench], ew_sig_high[cos_ssfr > quench]], 
                    ms=3.5, marker='s', capsize=4, ls='', c='b')
    ax[i].axhline(det_thresh[i], ls='--', c='k', lw=1)
    ax[i].set_xlabel('Impact parameter')
    ax[i].set_ylabel('EW ' + line)
    ax[i].set_ylim(-2.5, ylim)

    if (cos_survey == 'dwarfs') & (line == 'CIV1548'):
        plot_dwarfs_civ(ax[i])
    elif (line == 'H1215') & (cos_survey == 'dwarfs'):
        plot_dwarfs_lya(ax[i])
    elif (cos_survey == 'halos') & (line in halos_ions):
        plot_halos(ax[i], line)

    ax[i].legend(loc=1)


plt.savefig(plot_dir+'ions_impact_parameter.png')


