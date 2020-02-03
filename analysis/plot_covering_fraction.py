import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import sys
import numpy as np

sys.path.append('../cos_samples/')
from get_cos_info import get_cos_halos, get_cos_dwarfs, read_halos_data, get_cos_dwarfs_lya, get_cos_dwarfs_civ

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def compute_cfrac(ew, rho, rho_bins, det_thresh):
    digitized = np.digitize(rho, rho_bins)
    binned_ew = np.array([ew[digitized == i] for i in range(1, len(rho_bins))])
    binned_rho = np.array([rho[digitized == i] for i in range(1, len(rho_bins))])
    cfrac = np.zeros(len(binned_ew))
    for i in range(len(binned_ew)):
        if len(binned_ew[i]) == 0.:
            cfrac[i] = np.nan
        else:
            cfrac[i] = float((len(np.where(binned_ew[i] > det_thresh)[0]))) / float(len(binned_ew[i]))

    return np.array(cfrac)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

cos_survey = ['halos', 'dwarfs', 'halos', 'halos', 'dwarfs', 'halos']
lines = ['H1215', 'H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031']
plot_lines = [r'$\textrm{H}1215$', r'$\textrm{H}1215$', r'$\textrm{MgII}2796$',
                r'$\textrm{SiIII}1206$', r'$\textrm{CIV}1548$', r'$\textrm{OVI}1031$']
det_thresh = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1] # check CIV with Rongmon, check NeVIII with Jessica?

model = 'm100n1024'
wind = 's50'
mlim = np.log10(5.8e8) # lower limit of M*
ylim = 0.7
plot_dir = 'plots/'

rho_bins = np.arange(0., 200., 40.)
plot_bins = rho_bins[:-1] + 20

fig, ax = plt.subplots(3, 2, figsize=(12, 14))
ax = ax.flatten()

halo_rho, halo_M, halo_ssfr = get_cos_halos()
dwarfs_rho, dwarfs_M, dwarfs_ssfr = get_cos_dwarfs()

halos_rho_long = np.repeat(halo_rho, 20.)
dwarfs_rho_long = np.repeat(dwarfs_rho, 20.)


for i, survey in enumerate(cos_survey):

    cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        mass = np.repeat(f['mass'][:], 4)
        ssfr = np.repeat(f['ssfr'][:], 4)
    ssfr[ssfr < -11.5] = -11.5

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

    quench = -1.8  + 0.3*z - 9.

    if (survey == 'dwarfs') & (lines[i] == 'H1215'):
        cos_M = np.delete(cos_M, 3)
        cos_ssfr = np.delete(cos_ssfr, 3)
        cos_rho = np.delete(cos_rho, 3)

    cos_rho = cos_rho[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]

    cos_rho_long = np.repeat(cos_rho, 20)

    ew_file = 'data/cos_'+survey+'_'+model+'_'+snap+'_ew_data.h5'
    with h5py.File(ew_file, 'r') as f:
        ew = f[lines[i]+'_wave_ew'][:]

    # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
    if (survey == 'dwarfs') & (lines[i] == 'H1215'):
        ssfr = np.delete(ssfr, np.arange(3*20, 4*20))
        ew = np.delete(ew, np.arange(3*20, 4*20))

    cfrac = compute_cfrac(ew[ssfr > quench], cos_rho_long[ssfr > quench], rho_bins, det_thresh[i])
    ax[i].plot(plot_bins, cfrac, c='b', marker='o', ls='--')
    cfrac = compute_cfrac(ew[ssfr < quench], cos_rho_long[ssfr < quench], rho_bins, det_thresh[i])
    ax[i].plot(plot_bins, cfrac, c='r', marker='o', ls='--')
    
    if (survey == 'dwarfs') & (lines[i] == 'CIV1548'):
        EW, EWerr, EW_less_than = get_cos_dwarfs_civ() #in mA
        EW /= 1000.
    elif (survey == 'dwarfs') & (lines[i] == 'H1215'):
        EW, EWerr = get_cos_dwarfs_lya() # in mA
        EW /= 1000.
        EW = np.delete(EW, 3) # delete the measurements from Cos dwarfs galaxy 3 for the Lya stuff
    elif (survey == 'halos'):
        EW, EWerr = read_halos_data(lines[i])
        EW = np.abs(EW)

    EW = EW[cos_M > mlim]

    cfrac = compute_cfrac(EW[cos_ssfr > quench], cos_rho[cos_ssfr > quench], rho_bins, det_thresh[i])
    ax[i].plot(plot_bins, cfrac, c='c', marker='o', ls='--', label=label+' SF')
    cfrac = compute_cfrac(EW[cos_ssfr < quench], cos_rho[cos_ssfr < quench], rho_bins, det_thresh[i])
    ax[i].plot(plot_bins, cfrac, c='m', marker='o', ls='--', label=label+' Q')

    ax[i].set_xlabel(r'$\rho (\textrm{kpc})$')
    ax[i].set_ylabel(r'$\textrm{Covering fraction},\ $' + plot_lines[i])
    ax[i].set_ylim(0, 1.1)
    ax[i].legend(fontsize=10.5)

plt.savefig(plot_dir+'rho_cfrac.png')

