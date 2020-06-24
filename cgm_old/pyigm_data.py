import pyigm
from pyigm.cgm import cos_halos as pch


# Get data for igm systems from pyigm.
cos_halos = pch.COSHalos()
# If you downloaded a different version and have it in a tar.gz file, use, for example:
# cos_halos_v10 = pch.COSHalos(load=False)
# cos_halos_v10.load_sys(tfile='COS-Halos_sys.tar.gz')

# For example, get equivalent width data for the OVI 1031 line:
# Use the .keys() function to find your desired fields.
# I use this method to get all the data I need - it's not the easiest to interact with but it turned out to be easier
# to keep the data in dicts and have it all together rather than interact with different tables.
transition = 'OVI 1031'
for sys in cos_halos:
    sys = sys.to_dict()
    for component in sys['igm_sys']['components']:
        for line in sys['igm_sys']['components'][component]['lines']:
            tname = sys['igm_sys']['components'][component]['lines'][line]['name']
            if tname == transition:
                print sys['Name'], sys['igm_sys']['components'][component]['lines'][line]['attrib']['EW']['value']

# Alternatively, use a transition table for that transition. For example, get column densities for OVI 1031.
tbl = cos_halos.trans_tbl(transition)
print tbl
logN = tbl['logN']

# Or an ion table (gives different fields/data for the ion)
atomic_number = 8
ionization_number = 6
ion_tbl = cos_halos.ion_tbl((atomic_number, ionization_number))
print ion_tbl

# To get the updated Cloudy modeling data from Werk+14, load it in. For example, get the NHI data.
cos_halos.load_werk14()
for galaxy in cos_halos.werk14_cldy:
    print galaxy['GALID'], galaxy['NHI_BEST']
