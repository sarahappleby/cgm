# from Rongmon Bordoloi -
#   a data file with the HI measurements as in Bordoloi+2018 
#   42 entries as one galaxy has an intervening LLS system

from astropy.io import fits

data_file = fits.open('/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/COS-Dwarfs_Lya.fits')

data = data_file[1].data

# check that this is HI:
species = data['SPECIES']

# get equivalent widths:
EW = data['EW']
EWerr = data['EWERR']
