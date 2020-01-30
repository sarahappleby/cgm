import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
import sys
sys.path.append('../cos_samples/')
from get_cos_info import *

mlim = np.log10(5.8e8)

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def plot_dwarfs_lya(ax):

    EW, EWerr = get_cos_dwarfs_lya() # in mA
    EW /= 1000.
    EWerr /= 1000.
    EW, EWerr = convert_to_log(EW, EWerr)

    data_file = fits.open('/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/COS-Dwarfs_Lya.fits')
    data = data_file[1].data
    cos_M = data['SMASS'][0]
    cos_rho = data['RHO'][0] # in kpc

    ax.errorbar(cos_rho[cos_M > mlim], EW[cos_M > mlim], yerr=EWerr[cos_M > mlim], 
                c='k', ls='', marker='o', markersize=3, label='COS-Dwarfs')

def plot_dwarfs_civ(ax):

    cos_rho, cos_M, cos_ssfr = get_cos_dwarfs()

    EW, EWerr, EW_less_than = get_cos_dwarfs_civ() #in mA
    EW /= 1000.
    EWerr /= 1000.
    EW, EWerr = convert_to_log(EW, EWerr)


    cos_rho = cos_rho[cos_M >mlim]
    EW = EW[cos_M > mlim]
    EWerr = EWerr[cos_M > mlim]
    EW_less_than = EW_less_than[cos_M > mlim]
    cos_M = cos_M[cos_M > mlim]

    ax.errorbar(cos_rho[np.invert(EW_less_than)], EW[np.invert(EW_less_than)], yerr=EWerr[np.invert(EW_less_than)],
            c='k', ls='', marker='o', markersize=3, label='COS-Dwarfs')
    ax.scatter(cos_rho[EW_less_than], EW[EW_less_than], c='k', marker='$\downarrow$', ls='')

def plot_halos(ax, line):

    cos_rho, cos_M, cos_ssfr = get_cos_halos()
    EW, EWerr = get_cos_halos_lines(line)
    EW_upper_lim = (EW < 0.)
    EW, EWerr = convert_to_log(np.abs(EW), EWerr)

    ax.errorbar(cos_rho[np.invert(EW_upper_lim)], EW[np.invert(EW_upper_lim)], yerr=EWerr[np.invert(EW_upper_lim)],
            c='k', ls='', marker='o', markersize=3, label='COS-Halos')
    ax.errorbar(cos_rho[EW_upper_lim], EW[EW_upper_lim], yerr=EWerr[EW_upper_lim],
            c='k', marker='$\downarrow$', ls='')


    
