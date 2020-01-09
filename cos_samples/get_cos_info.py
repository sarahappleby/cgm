# from Rongmon Bordoloi -
#   a data file with the HI measurements as in Bordoloi+2018 
#   42 entries as one galaxy has an intervening LLS system

from astropy.io import fits, ascii
import h5py
import matplotlib.pyplot as plt
import numpy as np

def get_cos_dwarfs():
    table_file = '/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/line_table_simple.tex'
    table = ascii.read(table_file, format='latex')
    cos_rho = table['Rho']
    cos_M = table['logM_stellar']
    cos_ssfr = table['logsSFR']

    # identify galaxies with sSFR lower limit
    ssfr_less_than = np.array([False for i in list(range(len(cos_ssfr)))])
    ssfr_less_than[15] = True
    ssfr_less_than[16] = True
    ssfr_less_than[36:] = np.array([True for i in list(range(7))])

    for i, item in enumerate(cos_ssfr):
        if '$<$' in item:
            j = item.find('-')
            cos_ssfr[i] = item[j:]
    cos_ssfr = np.array(cos_ssfr, dtype=float) # SA
    return np.array(cos_rho), np.array(cos_M), cos_ssfr

def get_cos_halos():
    from pyigm.cgm import cos_halos as pch
    cos_halos = pch.COSHalos()
    cos_M = []
    cos_ssfr = []
    cos_rho = []
    for cos in cos_halos:
        cos = cos.to_dict()
        cos_M.append(cos['galaxy']['stellar_mass'])
        cos_ssfr.append(cos['galaxy']['ssfr'])
        cos_rho.append(cos['rho'])

    return np.array(cos_rho), np.array(cos_M), np.log10(cos_ssfr)

def get_cos_dwarfs_lya():
    data_file = fits.open('/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/COS-Dwarfs_Lya.fits')

    data = data_file[1].data

    # check that this is HI:
    species = data['SPECIES']

    # get equivalent widths:
    EW = data['EW']
    EWerr = data['EWERR']
