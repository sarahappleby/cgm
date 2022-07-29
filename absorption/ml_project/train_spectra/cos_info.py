import pandas as pd
import pyigm
from astropy.table import Table, Column

import glob
import os
import tarfile
import json
from collections import OrderedDict

lookup_data = {'pg_lines' : ['H1215', 'MgII2796', 'CII1334', 'SiIII1206', 'OVI1031'],
               'ions' : ['HI', 'MgII', 'CII', 'SiIII', 'OVI'],
               'lines' : ['1215.67', '2796.3543', '1334.5323', '1206.5', '1031.9261']}

pg_line = 'H1215'
place = lookup_data['pg_lines'].index(pg_line)
ion = lookup_data['ions'][place]
line = lookup_data['lines'][place]

features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot']

cdir = pyigm.__path__[0]+'/data/CGM/COS_Halos/'
survey = 'COS-Halos'
data_dir = os.getenv('COSHALOS_DATA')
fits_path = cdir+'/Summary/'
werk14_cldy = Table.read(fits_path+'coshaloscloudysol_newphi.fits')
kin_init_file = cdir+'/Kin/coshalo_kin_driver.dat'

tarfiles = glob.glob(cdir+'cos-halos_systems.v*.tar.gz')
tarfiles.sort()
tfile = tarfiles[-1]

_dict = OrderedDict()

tar = tarfile.open(tfile)
for i, member in enumerate(tar.getmembers()):
    if '.json' not in member.name:
        print('Skipping a likely folder: {:s}'.format(member.name))
        continue
    # Extract
    f = tar.extractfile(member)
    try:
        tdict = json.load(f)
    except:
        print('Unable to load {}'.format(member))
        continue
    # Build dict
    _dict[tdict['Name']] = tdict
tar.close()

data = pd.DataFrame(columns=features)

for i in range(len(werk14_cldy)):

    name = f"{werk14_cldy['FIELD'][i]}_{werk14_cldy['GALID'][i]}"
    cos = _dict[name]
  
    keys = list(cos['igm_sys']['components'].keys())
    end = keys[0].split('_')[1]
    
    
