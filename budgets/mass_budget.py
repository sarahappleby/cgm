import numpy as np 
import caesar
from pygadgetreader import readsnap

def get_bin_edges(x_min, x_max, dx):
	return np.arange(x_min, x_max+dx, dx)

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

cold_temp = 1.e5
hot_temp = 1.e6
ism_density = 0.13 # hydrogen number density, cm**-3
# pressurised ISM temperature floor
min_mass = 9.5
max_mass = 12.
dm = 0.5 # dex
mass_bins = get_bin_edges(min_mass, max_mass, dm)
plot_bins = get_bin_middle(mass_bins)

data_dir = '/home/sarah/data/'
snapfile = data_dir+'snap_m12.5n128_135.hdf5'
caesarfile = data_dir+'caesar_snap_m12.5n128_135.hdf5'

sim = caesar.load(caesarfile)
h = sim.simulation.hubble_constant

gal_sm = np.array([i.masses['stellar'].in_units('Msun') for i in sim.central_galaxies])
gal_sfr = np.array([i.sfr.in_units('Msun/yr') for i in sim.central_galaxies])
gal_ssfr = gal_sfr / gal_sm
gal_sm = np.log10(gal_sm)
gal_ssfr = np.log10(gal_ssfr)

gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K
#dust_frac = readsnap(snapfile, 'Dust_Masses', 'gas', suppress=1, units=1) # in fraction
#dust_z = readsnap(snapfile, 'Dust_Metallicity', 'gas', suppress=1, units=1)
star_mass = readsnap(snapfile, 'mass', 'star', suppress=1, units=1) / h # in Mo

phases = ['Cool CGM', 'Warm CGM', 'Hot CGM', 'ISM', 'Wind', 'Dust', 'Stars']
colours = ['m', 'tab:orange', 'g', 'b', 'c', 'tab:pink', 'r']
masses = np.zeros((7, len(sim.central_galaxies)))

for i in range(len(sim.central_galaxies)):
	glist = sim.galaxies[i].halo.glist
	slist = sim.galaxies[i].halo.slist # there can be stars in the halo of the galaxy that skews the stellar mass estimate of halo

	cgm_gas_mask = gas_nh[glist] < ism_density
	cold_gas_mask = gas_temp[glist] < cold_temp
	warm_gas_mask = (gas_temp[glist] > cold_temp) & (gas_temp[glist] < hot_temp)
	hot_gas_mask = gas_temp[glist] > hot_temp
	wind_mask = gas_delaytime[glist] > 0.

	masses[0][i] = np.sum(gas_mass[glist][cgm_gas_mask * cold_gas_mask]) # cold CGM gas
	masses[1][i] = np.sum(gas_mass[glist][cgm_gas_mask * warm_gas_mask]) # warm CGM gas
	masses[2][i] = np.sum(gas_mass[glist][cgm_gas_mask * hot_gas_mask]) # hot CGM gas
	masses[3][i] = np.sum(gas_mass[glist][np.invert(cgm_gas_mask)]) # ISM gas
	masses[4][i] = np.sum(gas_mass[glist][wind_mask]) # Wind particles
	#masses[5][i] = np.sum(gas_mass[glist]*dust_frac[glist]) # Dust particles
	masses[6][i] = np.sum(star_mass[slist]) # Stars

total_mass = np.sum(masses, axis=0)
mass_fracs = masses / total_mass

medians = np.zeros((len(phases), len(plot_bins)))
per_25 = np.zeros((len(phases), len(plot_bins)))
per_75 = np.zeros((len(phases), len(plot_bins)))
n_bin = np.zeros((len(phases), len(plot_bins)))

for i in range(len(phases)):
	binned_data = bin_data(gal_sm, masses[i], mass_bins)
	medians[i] = [np.nanpercentile(j, 50.) for j in binned_data]
	per_25[i] = [np.nanpercentile(j, 25.) for j in binned_data]
	per_75[i] = [np.nanpercentile(j, 75.) for j in binned_data]
	n_bin[i] = [len(j) for j in binned_data]

for i in range(len(phases)):
	plt.plot(plot_bins, np.log10(medians[i]), color=colours[i], label=phases[i])
plt.legend(loc=2)
plt.xlabel('M*')
plt.ylabel('Mass')
plt.xlim(min_mass, max_mass)
plt.savefig('/home/sarah/cgm/budgets/mass_mufasa_128.png')
plt.clf()

medians = np.zeros((len(phases), len(plot_bins)))
per_25 = np.zeros((len(phases), len(plot_bins)))
per_75 = np.zeros((len(phases), len(plot_bins)))
n_bin = np.zeros((len(phases), len(plot_bins)))

for i in range(len(phases)):
	binned_data = bin_data(gal_sm, mass_fracs[i], mass_bins)
	medians[i] = [np.nanpercentile(j, 50.) for j in binned_data]
	per_25[i] = [np.nanpercentile(j, 25.) for j in binned_data]
	per_75[i] = [np.nanpercentile(j, 75.) for j in binned_data]
	n_bin[i] = [len(j) for j in binned_data]

for i in range(len(phases)):
	plt.plot(plot_bins, medians[i], color=colours[i], label=phases[i])
plt.legend(loc=2)
plt.xlabel('M*')
plt.ylabel('Mass fraction')
plt.xlim(min_mass, max_mass)
plt.savefig('/home/sarah/cgm/budgets/mass_frac_mufasa_128.png')
plt.clf()