import numpy as np
import matplotlib.pyplot as plt
import caesar
import h5py
import sys

sys.path.append('/disk01/sapple/tools/')
from plotmedian import runningmedian

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


def chi_squared(data, model, noise):
    return np.sum((data - model)**2 / noise**2)


def reduced_chi_squared(data, model, noise, dof):
    return chi_squared(data, model, noise) / dof


def plot_model_fit(mhalo, mstar, lower, upper, mstar_model, coeffs, filename):
    plt.errorbar(mhalo, mstar, yerr=[lower, upper], label='data')
    plt.plot(mhalo, mstar_model, label='model')
    plt.legend()
    plt.title(f'Coefficients: {repr(coeffs)}')
    plt.xlabel(r'${\rm log} (M_{\rm halo}/M_{\odot})$')
    plt.ylabel(r'${\rm log} (M_\star/M_{\odot})$')
    plt.savefig(filename)
    plt.clf()


def get_mhalo_mstar_fit(sim, mstar_min=9., order=1):
    gal_mstar = np.log10(np.array([i.masses['stellar'].in_units('Msun') for i in sim.galaxies]))
    gal_mhalo = np.log10(np.array([i.halo.masses['total'].in_units('Msun') for i in sim.galaxies]))
    gal_cent = np.array([i.central for i in sim.galaxies])
    mstar_mask = gal_mstar > mstar_min

    mhalo_bins, median, lower, upper, ngals = runningmedian(gal_mhalo[gal_cent*mstar_mask], gal_mstar[gal_cent*mstar_mask])
    
    mhalo_bins = mhalo_bins[~np.isnan(median)]
    lower = lower[~np.isnan(median)]
    upper = upper[~np.isnan(median)]
    median = median[~np.isnan(median)]

    dof = len(mhalo_bins) - order
    coeffs = np.polyfit(mhalo_bins, median, order)

    model_vals = np.polyval(coeffs, mhalo_bins)
    chi_r = reduced_chi_squared(median, model_vals, np.minimum(lower, upper), dof)
    print(f'Reduced Chi Squared = {chi_r}')
    print(f'Coefficients: {coeffs}')

    plot_model_fit(mhalo_bins, median, lower, upper, model_vals, np.round(coeffs, 4), f'plots/{model}_{wind}_{snap}_mhalo_mstar_deg_{order}.png')
    return coeffs


def get_mhalo_axis_values(min_mhalo=11., max_mhalo=15., dmhalo=1.0, model='m100n1024', wind='s50'):
    if (model == 'm100n1024') & (wind == 's50'):
        coeffs=np.array([0.739, 1.229])
    elif (model == 'm50n512') & (wind in ['s50', 's50j7k']):
        coeffs=np.array([0.804, 0.403])
    elif (model == 'm50n512') & (wind == 's50nox'):
        coeffs=np.array([0.885, -0.397])
    elif (model == 'm50n512') & (wind == 's50nojet'):
        coeffs=np.array([1.06, -2.281])
    elif (model == 'm50n512') & (wind == 's50nofb'):
        coeffs=np.array([0.902, -0.192])
    mhalo = np.arange(min_mhalo, max_mhalo+dmhalo, dmhalo)
    mstar = np.polyval(coeffs, mhalo)
    return mhalo, mstar


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    order = int(sys.argv[4])

    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')

    coeffs = get_mhalo_mstar_fit(sim, order=order)
