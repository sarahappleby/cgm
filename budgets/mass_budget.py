import numpy as np
import h5py 
import caesar
from pygadgetreader import readsnap


photo_temp = 10.**4.5 # in K
cold_temp = 1.e5
hot_temp = 1.e6
ism_density = 0.13 # hydrogen number density, cm**-3
omega_b = 0.048
omega_m = 0.3
f_baryon = omega_b / omega_m

data_dir = '/home/sarah/data/'
snapfile = data_dir+'snap_m12.5n128_135.hdf5'
caesarfile = data_dir+'caesar_snap_m12.5n128_135.hdf5'
savedir = '/home/sarah/cgm/budgets/data/'

sim = caesar.load(caesarfile)
h = sim.simulation.hubble_constant

gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.central_galaxies])
gal_sfr = np.array([i.sfr.in_units('Msun/yr') for i in sim.central_galaxies])
gal_tvir = np.array([i.halo.virial_quantities['temperature'].in_units('K') for i in sim.central_galaxies])
gal_ssfr = gal_sfr / gal_sm
gal_sm = np.log10(gal_sm)
gal_ssfr = np.log10(gal_ssfr)

dm_mass = readsnap(snapfile, 'mass', 'dm', suppress=1, units=1) / h # in Mo)
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
		  'ISM', 'Wind', 'Dust', 'Stars', 'Dark matter', 'Total baryons']
mass_budget = {phase: np.zeros(len(sim.central_galaxies)) for phase in phases}
metal_budget = {phase: np.zeros(len(sim.central_galaxies)) for phase in phases}
del metal_budget['Dark matter']

# get the particle ids for different gas phases
all_gas_ids = np.arange(len(gas_mass))
gparts = {phase: np.array([]) for phase in phases}
del gparts['Dust'], gparts['Stars'], gparts['Dark matter'], gparts['Total baryons']

for i in range(len(sim.central_galaxies)):
	glist = sim.galaxies[i].halo.glist
	slist = sim.galaxies[i].halo.slist
	dmlist = sim.galaxies[i].halo.dmlist

	cgm_gas_mask = gas_nh[glist] < ism_density
	wind_mask = gas_delaytime[glist] > 0.

	cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < cold_temp)
	warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > cold_temp) & (gas_temp[glist] < hot_temp)
	hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > hot_temp)
	gparts['Cool CGM (T < 10^5)'] = np.append(gparts['Cool CGM (T < 10^5)'], all_gas_ids[glist][cool_gas_mask])
	gparts['Warm CGM (10^5 < T < 10^6)'] = np.append(gparts['Warm CGM (10^5 < T < 10^6)'], all_gas_ids[glist][warm_gas_mask])
	gparts['Hot CGM (T > 10^6)'] = np.append(gparts['Hot CGM (T > 10^6)'], all_gas_ids[glist][hot_gas_mask])
	mass_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
	mass_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
	mass_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
	metal_budget['Cool CGM (T < 10^5)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
	metal_budget['Warm CGM (10^5 < T < 10^6)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
	metal_budget['Hot CGM (T > 10^6)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])

	cool_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] < photo_temp)
	warm_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > photo_temp) & (gas_temp[glist] < gal_tvir[i])
	hot_gas_mask = cgm_gas_mask & np.invert(wind_mask) & (gas_temp[glist] > gal_tvir[i])
	gparts['Cool CGM (T < Tphoto)'] = np.append(gparts['Cool CGM (T < Tphoto)'], all_gas_ids[glist][cool_gas_mask])
	gparts['Warm CGM (Tphoto < T < Tvir)'] = np.append(gparts['Warm CGM (Tphoto < T < Tvir)'], all_gas_ids[glist][warm_gas_mask])
	gparts['Hot CGM (T > Tvir)'] = np.append(gparts['Hot CGM (T > Tvir)'], all_gas_ids[glist][hot_gas_mask])
	mass_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask])
	mass_budget['Warm CGM (Tphoto < T < Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask])
	mass_budget['Hot CGM (T > Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask])
	metal_budget['Cool CGM (T < Tphoto)'][i] = np.sum(gas_mass[glist][cool_gas_mask] * gas_z[glist][cool_gas_mask])
	metal_budget['Warm CGM (Tphoto < T < Tvir)'][i] = np.sum(gas_mass[glist][warm_gas_mask] * gas_z[glist][warm_gas_mask])
	metal_budget['Hot CGM (T > Tvir)'][i] = np.sum(gas_mass[glist][hot_gas_mask] * gas_z[glist][hot_gas_mask])

	mass_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
	mass_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask])
	gparts['ISM'] = np.append(gparts['ISM'], all_gas_ids[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
	gparts['Wind'] = np.append(gparts['Wind'], all_gas_ids[glist][wind_mask])
	mass_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)])
	mass_budget['Stars'][i] = np.sum(star_mass[slist])
	mass_budget['Dark matter'][i] = np.sum(dm_mass[dmlist])
	metal_budget['ISM'][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)] * gas_z[glist][np.invert(cgm_gas_mask) & np.invert(wind_mask)])
	metal_budget['Wind'][i] = np.sum(gas_mass[glist][wind_mask] * gas_z[glist][wind_mask])
	metal_budget['Dust'][i] = np.sum(dust_mass[glist][np.invert(wind_mask)] * dust_z[glist][np.invert(wind_mask)])
	metal_budget['Stars'][i] = np.sum(star_mass[slist] * star_z[slist])

	mass_budget['Total baryons'][i] = np.sum(gas_mass[glist]) + np.sum(dust_mass[glist]) + np.sum(star_mass[slist])
	metal_budget['Total baryons'][i] = np.sum(gas_mass[glist] * gas_z[glist]) + np.sum(dust_mass[glist] * dust_z[glist]) + np.sum(star_mass[slist] * star_z[slist])

gparts = {k: np.unique(np.sort(p.astype('int'))) for k, p in gparts.items()}
dmparts = np.concatenate(np.array([i.halo.dmlist for i in sim.central_galaxies]))
dmparts = np.unique(np.sort(dmparts))
sparts = np.concatenate(np.array([i.halo.slist for i in sim.central_galaxies]))
sparts = np.unique(np.sort(sparts))

available_mass_fractions = {k: p/mass_budget['Total baryons'] for k, p in mass_budget.items()} 
available_mass_fractions['Total baryons'] = mass_budget['Total baryons'].copy()
del available_mass_fractions['Dark matter']

cosmic_baryons = (mass_budget['Total baryons'] + mass_budget['Dark matter']) * f_baryon
omega_mass_fractions = {k: p/cosmic_baryons for k, p in mass_budget.items()}
omega_mass_fractions['Cosmic baryon mass'] = cosmic_baryons.copy()
del omega_mass_fractions['Dark matter']

with h5py.File(savedir+'mass_budget.h5', 'a') as hf: 
	for k, p in mass_budget.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'available_mass_fraction.h5', 'a') as hf: 
	for k, p in available_mass_fractions.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'omega_mass_fraction.h5', 'a') as hf: 
	for k, p in omega_mass_fractions.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'metal_budget.h5', 'a') as hf: 
	for k, p in metal_budget.items(): 
		hf.create_dataset(k, data=np.array(p)) 

with h5py.File(savedir+'particle_ids.h5', 'a') as hf:
	for k, p in gparts.items():
		hf.create_dataset(k, data=np.array(p))
	hf.create_dataset('Stars', data=np.array(sparts))
	hf.create_dataset('Dark matter', data=np.array(dmparts))