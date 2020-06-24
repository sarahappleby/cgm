import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
import h5py
import os
import sys
sys.path.append('../cos_samples/')
from get_cos_info import get_cos_dwarfs, get_cos_halos, get_cos_dwarfs_lya, get_cos_dwarfs_civ, read_halos_data

mlim = np.log10(5.8e8)
h = 0.68

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def plot_dwarfs_lya(ax, quench, r200_scaled=False):

    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()

    if r200_scaled:
        cos_rho = cos_rho.astype(float) * h
        dist = cos_rho / cos_r200
    else:
        dist = cos_rho.copy()

    EW, EWerr = get_cos_dwarfs_lya() # in mA
    EW /= 1000.
    EWerr /= 1000.
    EW, EWerr = convert_to_log(EW, EWerr)

    dist = np.delete(dist, 3)
    cos_ssfr = np.delete(cos_ssfr, 3)
    cos_M = np.delete(cos_M, 3)
    EW = np.delete(EW, 3)
    EWerr = np.delete(EWerr, 3)

    dist = dist[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    EW = EW[cos_M > mlim]
    EWerr = EWerr[cos_M > mlim]

    c1 = ax.errorbar(dist[cos_ssfr > quench], EW[cos_ssfr > quench], yerr=EWerr[cos_ssfr > quench],
                c='c', ls='', marker='x', markersize=6, capsize=4, label='COS-Dwarfs SF')
    c2 = ax.errorbar(dist[cos_ssfr < quench], EW[cos_ssfr < quench], yerr=EWerr[cos_ssfr < quench],
                c='m', ls='', marker='x', markersize=6, capsize=4, label='COS-Dwarfs Q')
    return c1, c2

def plot_dwarfs_civ(ax, quench, r200_scaled=False):

    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()

    if r200_scaled:
        cos_rho = cos_rho.astype(float)* h
        dist = cos_rho / cos_r200
    else:
        dist = cos_rho.copy()

    EW, EWerr, EW_less_than = get_cos_dwarfs_civ() #in mA
    EW /= 1000.
    EWerr /= 1000.
    EW, EWerr = convert_to_log(EW, EWerr)

    dist = dist[cos_M >mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    EW = EW[cos_M > mlim]
    EWerr = EWerr[cos_M > mlim]
    EW_less_than = EW_less_than[cos_M > mlim]
    cos_M = cos_M[cos_M > mlim]

    mask = np.invert(EW_less_than) * (cos_ssfr > quench)
    c1 = ax.errorbar(dist[mask], EW[mask], yerr=EWerr[mask], 
                c='c', ls='', marker='x', markersize=6, label='COS-Dwarfs SF')
    mask = np.invert(EW_less_than) * (cos_ssfr < quench)
    c2 = ax.errorbar(dist[mask], EW[mask], yerr=EWerr[mask],
                c='m', ls='', marker='x', markersize=6, label='COS-Dwarfs Q')
    mask = EW_less_than * (cos_ssfr > quench)
    ax.scatter(dist[mask], EW[mask], c='c', marker='$\downarrow$', s=60.)
    mask = EW_less_than * (cos_ssfr < quench)
    ax.scatter(dist[mask], EW[mask], c='m', marker='$\downarrow$', s=60.)

    return c1, c2

def plot_halos(ax, line, quench, r200_scaled=False):

    z = 0.2
    cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_halos()
   
    if r200_scaled:
        cos_rho = cos_rho.astype(float) * h * (1+z)
        dist = cos_rho / cos_r200
    else:
        dist = cos_rho.copy()

    EW, EWerr = read_halos_data(line)
    
    EW_upper_lim = (EW < 0.)
    EW, EWerr = convert_to_log(np.abs(EW), EWerr)

    mask = np.invert(EW_upper_lim) * (cos_ssfr > quench)
    c1 = ax.errorbar(dist[mask], EW[mask], yerr=EWerr[mask],
            c='c', ls='', marker='x', markersize=6, capsize=4, label='COS-Halos SF')
    mask = np.invert(EW_upper_lim) * (cos_ssfr < quench)
    c2 = ax.errorbar(dist[mask], EW[mask], yerr=EWerr[mask],
            c='m', ls='', marker='x', markersize=6, capsize=4, label='COS-Halos Q')
    mask = EW_upper_lim * (cos_ssfr > quench)
    ax.scatter(dist[mask], EW[mask],
            c='c', marker='$\downarrow$', s=60.)
    mask = EW_upper_lim * (cos_ssfr < quench)
    ax.scatter(dist[mask], EW[mask],
            c='m', marker='$\downarrow$', s=60.)

    return c1, c2
    
