# Some galaxies in the sample had zeros for their r200 so needed to replace these :(
# The current pipeline filters out these galaxies before the selection.

import caesar
import yt
import numpy as np
import h5py
import sys

def belfiore_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sfms_line(mstar, a=1., b=-10.8):
    # a cut of -10.8/yr as in other Simba work
    return mstar*a + b

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask

# Randomly select ngals_each galaxies in each region
def choose_gals(gal_ids, ngals_each, seed=1):
    if seed is not None:
        np.random.seed(seed)
    ngals = len(gal_ids)
    chosen = np.random.choice(ngals, ngals_each, replace=False)
    return gal_ids[chosen]

model = 'm100n1024'
wind = 's50'
snap = '151'

delta_m = 0.25
min_m = 10.
nbins_m = 6
mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
ngals_each = 12
nbins_ssfr = 3

sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
data_dir = f'/home/rad/data/{model}/{wind}/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
co = yt.utilities.cosmology.Cosmology(hubble_constant=sim.simulation.hubble_constant,
                                      omega_matter=sim.simulation.omega_matter,
                                      omega_lambda=sim.simulation.omega_lambda)
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift
quench = quench_thresh(redshift)

gal_cent = np.array([i.central for i in sim.galaxies])
gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_lgas = yt.YTArray([sim.galaxies[i].rotation['gas_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_lbaryon = yt.YTArray([sim.galaxies[i].rotation['baryon_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

halo_r200 = np.array([i.halo.virial_quantities['r200c'].in_units('kpc/h') for i in sim.galaxies]) 
halo_pos = np.array([i.halo.pos.in_units('kpc/h') for i in sim.galaxies])
r200_mask = halo_r200 > 0.

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf: 
    gal_ids = sf['gal_ids'][:]
    mass = sf['mass'][:]
    ssfr = sf['ssfr'][:]
    sfr = sf['sfr'][:]
    fgas = sf['fgas'][:]
    pos = sf['position'][:]
    vgal_pos = sf['vgal_position'][:]
    lgas = sf['L_gas'][:]
    lbar = sf['L_baryon'][:]
    h_r200 = sf['halo_r200'][:]
    h_pos = sf['halo_pos'][:]


replace_ids = np.where(h_r200 == 0.)[0]
mass_binned = np.digitize(mass[replace_ids],mass_bins) -1

available = np.array([True for i in range(len(sim.galaxies))])
available[gal_ids] = False
new_gals = np.zeros(len(replace_ids), dtype=int)

for i in range(len(replace_ids)):
    
    mass_mask = (gal_sm > mass_bins[mass_binned[i]]) & (gal_sm < mass_bins[mass_binned[i]]+delta_m)
    if ssfr[replace_ids[i]] > quench:
        ssfr_mask = gal_ssfr > quench
    elif (ssfr[replace_ids[i]] < quench) & (ssfr[replace_ids[i]] > (quench - 1.)):
        ssfr_mask = (gal_ssfr > (quench -1.)) & (gal_ssfr < quench)
    else:
        ssfr_mask = (gal_ssfr == -14.)

    mask = mass_mask * gal_cent * ssfr_mask * r200_mask * available
    new_gals[i] = np.random.choice(np.arange(len(sim.galaxies))[mask], size=1)

    available[new_gals[i]] = False

gal_ids[replace_ids] = new_gals
mass[replace_ids] = gal_sm[new_gals]
ssfr[replace_ids] = gal_ssfr[new_gals]
sfr[replace_ids] = gal_sfr[new_gals]
fgas[replace_ids] = gal_gas_frac[new_gals]
pos[replace_ids] = gal_pos[new_gals]
vgal_pos[replace_ids] = gal_vgal_pos[new_gals]
lgas[replace_ids] = gal_lgas[new_gals]
lbar[replace_ids] = gal_lbaryon[new_gals]
h_r200[replace_ids] = halo_r200[new_gals]
h_pos[replace_ids] = halo_pos[new_gals]


with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as hf:
    for key in list(hf.keys()):
        del hf[key]
    hf.create_dataset('gal_ids', data=np.array(gal_ids))
    hf.create_dataset('mass', data=np.array(mass))
    hf.create_dataset('ssfr', data=np.array(ssfr))
    hf.create_dataset('sfr', data=np.array(sfr))
    hf.create_dataset('fgas', data=np.array(fgas))
    hf.create_dataset('position', data=np.array(pos))
    hf.create_dataset('vgal_position', data=np.array(vgal_pos))
    hf.create_dataset('L_gas', data=np.array(lgas))
    hf.create_dataset('L_baryon', data=np.array(lbar))
    hf.create_dataset('halo_r200', data=np.array(h_r200))
    hf.create_dataset('halo_pos', data=np.array(h_pos))

