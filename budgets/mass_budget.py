import numpy as np
import h5py 
import caesar
from pygadgetreader import readsnap


photo_temp = 10.**4.5 # in K
cold_temp = 1.e5
hot_temp = 1.e6
ism_density = 0.13 # hydrogen number density, cm**-3

data_dir = '/home/sarah/data/'
snapfile = data_dir+'snap_m12.5n128_135.hdf5'
caesarfile = data_dir+'caesar_snap_m12.5n128_135.hdf5'
savedir = '/home/sarah/cgm/budgets/'

sim = caesar.load(caesarfile)
h = sim.simulation.hubble_constant

gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.central_galaxies])
gal_sfr = np.array([i.sfr.in_units('Msun/yr') for i in sim.central_galaxies])
gal_tvir = np.array([i.halo.virial_quantities['temperature'].in_units('K') for i in sim.central_galaxies])
gal_ssfr = gal_sfr / gal_sm
gal_sm = np.log10(gal_sm)
gal_ssfr = np.log10(gal_ssfr)

gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
gas_z = readsnap(snapfile, 'z', 'gas', suppress=1, units=1)
gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
#dust_mass = readsnap(snapfile, 'Dust_Masses', 'gas', suppress=1, units=1) / h # in Mo
#dust_z = readsnap(snapfile, 'Dust_Metallicity', 'gas', suppress=1, units=1)
dust_mass = np.zeros(len(gas_mass))
dust_z = np.zeros(len(gas_mass))
star_mass = readsnap(snapfile, 'mass', 'star', suppress=1, units=1) / h # in Mo
star_z = readsnap(snapfile, 'z', 'star', suppress=1, units=1)

phases = ['Cool CGM (T < 10^5)', 'Warm CGM (10^5 < T < 10^6)', 'Hot CGM (T > 10^6)',
		  'Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)', 
		  'ISM', 'Wind', 'Dust', 'Stars', 'Total']
mass_budget = {phase: np.zeros(len(sim.central_galaxies)) for phase in phases}
metal_budget = {phase: np.zeros(len(sim.central_galaxies)) for phase in phases}

for i in range(len(sim.central_galaxies)):
	glist = sim.galaxies[i].halo.glist
	slist = sim.galaxies[i].halo.slist

	cgm_gas_mask = gas_nh[glist] < ism_density
	wind_mask = gas_delaytime[glist] > 0.

	cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < cold_temp)
	warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > cold_temp) & (gas_temp[glist] < hot_temp)
	hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > hot_temp)
	mass_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
	mass_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
	mass_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
	metal_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
	metal_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
	metal_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])

	cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < photo_temp)
	warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > photo_temp) & (gas_temp[glist] < gal_tvir[i])
	hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > gal_tvir[i])	
	mass_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
	mass_budget['Warm CGM (Tphoto < T < Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
	mass_budget['Hot CGM (T > Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
	metal_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
	metal_budget['Warm CGM (Tphoto < T < Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
	metal_budget['Hot CGM (T > Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])

	mass_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
	mass_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask])
	mass_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)])
	mass_budget['Stars'][i] = np.sum(star_mass[slist])
	metal_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)] * gas_z[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
	metal_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask] * gas_z[glist][wind_mask])
	metal_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)] * dust_z[glist][np.invert(wind_mask)])
	metal_budget['Stars'][i] = np.sum(star_mass[slist] * star_z[slist])

	mass_budget['Total'][i] = np.sum(gas_mass[glist]) + np.sum(dust_mass[glist]) + np.sum(star_mass[slist])
	metal_budget['Total'][i] = np.sum(gas_mass[glist] * gas_z[glist]) + np.sum(dust_mass[glist] * dust_z[glist]) + np.sum(star_mass[slist] * star_z[slist])

mass_fractions = {k: p/mass_budget['Total'] for k, p in mass_budget.items()} 
mass_fractions['Total'] = mass_budget['Total'].copy()

with h5py.File(savedir+'mass_budget.h5', 'a') as hf: 
	for k, p in mass_budget.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'mass_fractions.h5', 'a') as hf: 
	for k, p in mass_fractions.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'metal_budget.h5', 'a') as hf: 
	for k, p in metal_budget.items(): 
		hf.create_dataset(k, data=np.array(p)) 