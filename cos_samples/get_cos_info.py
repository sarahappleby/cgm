# from Rongmon Bordoloi -
#   a data file with the HI measurements as in Bordoloi+2018 
#   42 entries as one galaxy has an intervening LLS system
#   only 38 are above the mass resolution limit

from astropy.io import fits, ascii
import h5py
import matplotlib.pyplot as plt
import numpy as np

def get_cos_dwarfs():
    table_file = '/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/line_table_simple.tex'
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

def get_cos_dwarfs_civ():
    table_file = '/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/line_table_simple.tex'
    table = ascii.read(table_file, format='latex')
    ew_civ = np.array(table['EW_CIV'])
    ew_civ_err = np.zeros(len(ew_civ))

    for i, item in enumerate(ew_civ):
        if '$<$' in item:
            ew_civ[i] = item[-2:]
        elif '$\\pm$' in item:
            stuff = item.split('$\\pm$') 
            ew_civ[i] = stuff[0]
            ew_civ_err[i] = stuff[1]
    ew_civ = np.array(ew_civ, dtype=float)
    ew_civ_err = np.array(ew_civ_err, dtype=float)

    ew_less_than = np.array([False for i in list(range(len(ew_civ)))])
    ew_less_than[ew_civ_err == 0.] = True
   
    return ew_civ, ew_civ_err, ew_less_than

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

def get_cos_halos_lines(pg_line):

    lookup_data = {'pg_lines' : ['H1215', 'MgII2796', 'SiIII1206', 'OVI1031'],
                    'ions' : ['HI', 'MgII', 'SiIII', 'OVI'], 
                    'lines' : [1215.67, 2796.3543, 1206.5, 1031.9261]}
    place = lookup_data['pg_lines'].index(pg_line)

    ion = lookup_data['ions'][place]
    line = lookup_data['lines'][place]

    from pyigm.cgm import cos_halos as pch
    cos_halos = pch.COSHalos()
    EW = []
    EWerr = []
    for i, cos in enumerate(cos_halos):
        cos = cos.to_dict()
        keys = list(cos['igm_sys']['components'].keys())
        end = keys[0].split('_')[1]
        try:
            data = cos['igm_sys']['components'][ion+'_'+end]['lines'][line]['attrib']
            EW.append(data['EW']['value'])
            EWerr.append(data['sig_EW']['value'])
        except KeyError:
            print('ion in keys but keyerror for '+str(i))
            EW.append(np.nan)
            EWerr.append(np.nan)
            continue
    return np.array(EW), np.array(EWerr)

def get_cos_dwarfs_lya():
    data_file = fits.open('/home/sapple/cgm/cos_samples/obs_data/cos_dwarfs/COS-Dwarfs_Lya.fits')

    data = data_file[1].data

    # check that this is HI:
    species = data['SPECIES']

    # get equivalent widths:
    EW = data['EW'][0]
    EWerr = data['EWERR'][0]

    return EW, EWerr
