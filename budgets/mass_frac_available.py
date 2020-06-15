import matplotlib.pyplot as plt
import numpy as np
import h5py

def get_bin_edges(x_min, x_max, dx):
	return np.arange(x_min, x_max+dx, dx)

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

min_mass = 9.5
max_mass = 12.
dm = 0.5 # dex
mass_bins = get_bin_edges(min_mass, max_mass, dm)
plot_bins = get_bin_middle(mass_bins)

data_dir = '/home/sarah/cgm/budgets/'
phases = ['Cool CGM (T < Tphoto)', 'Warm CGM (Tphoto < T < Tvir)', 'Hot CGM (T > Tvir)', 
		  'ISM', 'Wind', 'Dust', 'Stars']
colours = ['m', 'tab:orange', 'g', 'b', 'c', 'tab:pink', 'r']

mass_fractions = {}
with h5py.File(data_dir+'mass_fractions.h5', 'r') as hf:
	for p in phases:
		mass_budget[k] = hf[p][:]

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
