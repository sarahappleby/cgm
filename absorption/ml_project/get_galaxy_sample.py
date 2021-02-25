# Select 12 Simba galaxies in each of our SFR-Mstar bins
# Mask out the regions we want, using the definition of SF/GV/Q galaxies from Belfiore+18 and Appleby+20
# Save the galaxy properties for our sample galaxies

import caesar
import yt
import numpy as np
import h5py
import sys
from numba import jit

def sfms_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sf_check(mstar, sfr):
    # Check whether galaxy is star forming according to Belfiore+18
    sfr_lower = sfms_line(mstar, a=0.73, b=-7.7)
    if sfr_lower < sfr:
        return True
    else:
        return False

def gv_check(mstar, sfr):
    # Check whether galaxy is in the green valley according to Belfiore+18
    sfr_upper = sfms_line(mstar, a=0.73, b=-7.7)
    sfr_lower = sfms_line(mstar, a=0.73, b=-8.7)
    if (sfr_upper > sfr) & (sfr_lower < sfr):
        return True
    else:
        return False 

def q_check(mstar, sfr):
    # Check whether galaxy is quenched according to Belfiore+18
    sfr_upper = sfms_line(mstar, a=0.73, b=-8.7)
    if sfr_upper > sfr:
        return True
    else:
        return False

def ssfr_type_check(mstar, sfr):

    sf_line = sfms_line(mstar, a=0.73, b=-7.7)
    q_line = sfms_line(mstar, a=0.73, b=-8.7)

    sf_mask = sfr > sf_line
    gv_mask = (sfr < sf_line) & (sfr > q_line)
    q_mask = sfr < q_line
    return sf_mask, gv_mask, q_mask

# Randomly select ngals_each galaxies in each region
def choose_gals(gal_ids, ngals_each):
    ngals = len(gal_ids)
    chosen = np.random.choice(ngals, ngals_each)
    return gal_ids[chosen]

model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

delta_m = 0.25
min_m = 10.
nbins_m = 6
mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)
ngals_each = 12
nbins_ssfr = 3

save_dir = f'/disk01/sapple/cgm/absorption/ml_project/'
data_dir = f'/home/rad/data/{model}/{wind}/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift

gal_cent = np.array([i.central for i in sim.galaxies])
gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])
halo_r200 = np.array([sim.galaxies[int(i)].halo.virial_quantities['r200c'].in_units('kpc/h') for i in gal_ids]

# empty arrays to store ngals_each simba galaxies per ssfr-mstar bin
gal_ids = np.ones(nbins_m*nbins_ssfr*ngals_each) * np.nan

sf_mask, gv_mask, q_mask = ssfr_type_check(gal_sm, np.log10(gal_sfr))

for i in range(len(mass_bins)):
    gal_id_range = range((i)*ngals_each*nbins_ssfr, (i+1)*ngals_each*nbins_ssfr)
    mass_mask = (gal_sm > mass_bins[i]) & (gal_sm < (mass_bins[i] + delta_m))

    # get the star forming galaxies:
    mask = gal_cent * mass_mask * sf_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    gal_ids[gal_id_range[0:ngals_each]] = choose_gals(gals_possible, ngals_each)

    # get the green valley galaxies:
    mask = gal_cent * mass_mask * gv_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    gal_ids[gal_id_range[ngals_each:2*ngals_each]] = choose_gals(gals_possible, ngals_each)

    # get the quenched galaxies:
    mask = gal_cent * mass_mask * q_mask
    gals_possible = np.arange(len(sim.galaxies))[mask]
    gal_ids[gal_id_range[2*ngals_each:3*ngals_each]] = choose_gals(gals_possible, ngals_each)

gal_ids = gal_ids.astype('int')
halo_r200 = np.array([sim.galaxies[i].halo.virial_quantities['r200c'].in_units('kpc/h') for i in gal_ids])
halo_pos = np.array([sim.galaxies[i].halo.pos.in_units('kpc/h') for i in gal_ids])

with h5py.File(f'{save_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as hf:
    hf.create_dataset('gal_ids', data=np.array(gal_ids))
    hf.create_dataset('mass', data=np.array(gal_sm[gal_ids]))
    hf.create_dataset('ssfr', data=np.array(gal_ssfr[gal_ids]))
    hf.create_dataset('gas_frac', data=np.array(gal_gas_frac[gal_ids]))
    hf.create_dataset('position', data=np.array(gal_pos[gal_ids]))
    hf.create_dataset('vgal_position', data=np.array(gal_vgal_pos[gal_ids]))
    hf.create_dataset('halo_r200', data=np.array(halo_r200))
    hf.create_dataset('halo_pos', data=np.array(halo_pos))
    hf.attrs['pos_units'] = 'kpc/h'
    hf.attrs['mass_units'] = 'Msun'
    hf.attrs['ssfr_units'] = 'yr^-1'
    hf.attrs['sfr_units'] = 'Msun/yr'
    hf.attrs['vel_units'] = 'km/s'

