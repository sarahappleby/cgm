# Produce radial profiles of MgII metallicity for the sample galaxies

import numpy as np
import h5py
import os
import caesar
from pygadgetreader import readsnap
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.


cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'


if __name__ == '__main__':

    model = 'm100n1024'
    #model = 'm25n256'
    wind = 's50'
    snap = '151'

    delta_r = 0.125
    min_r = 0.
    max_r = 1.5
    rbins = np.arange(min_r, max_r, delta_r)
    plot_bins = get_bin_middle(rbins)

    delta_m = 0.25
    min_m = 10.
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)

    mass_titles = []
    for i in range(nbins_m):
        mass_titles.append(f'{mass_bins[i]}'+ r'$ < {\rm log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    data_dir = f'/home/rad/data/{model}/{wind}/'
    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'

    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    h = sim.simulation.hubble_constant
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)

    profile_file = f'/disk04/sapple/cgm/absorption/ml_project/data/profiles/{model}_{wind}_{snap}_profile.h5'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:

        gal_ids = sf['gal_ids'][:]
        gal_ssfr = sf['ssfr'][:]
        gal_mass = sf['mass'][:]
        gal_pos = sf['position'][:]
        gal_r200 = sf['halo_r200'][:]

    if not os.path.isfile(profile_file):

        gas_pos = readsnap(snapfile, 'pos', 'gas', suppress=1, units=1) / (1.+redshift) # in ckpc/h
        gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
        gas_mg = readsnap(snapfile, 'Metallicity', 'gas', suppress=1, units=1)[:, 6] / 7.14e-4 # in solar abundances
        gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
        wind_mask = gas_delaytime == 0.

        profiles = np.zeros((len(gal_ids), len(rbins) -1))

        for i in range(len(gal_ids)):

            glist = sim.galaxies[gal_ids[i]].halo.glist
            dist = np.linalg.norm(gas_pos - gal_pos[i], axis=1)
            r200 = gal_r200[i]

            for j in range(len(rbins) - 1):

                pos_mask = (dist > r200*rbins[j]) & (dist < r200*rbins[j+1])
                profiles[i][j] = np.median(gas_mg[pos_mask & wind_mask])

        with h5py.File(profile_file, 'a') as hf:
            hf.create_dataset('profile_Mg', data=np.array(profiles))

    else:
        
        with h5py.File(profile_file, 'r') as hf:
            profiles = hf['profile_Mg'][:]
            if profiles.shape == (212, 12):
                profiles = profiles[:, :-1]

    sf_mask = (gal_ssfr > quench)
    gv_mask = (gal_ssfr < quench) & (gal_ssfr > quench-1)
    q_mask = gal_ssfr == -14

    fig, ax = plt.subplots(1, len(mass_titles), figsize=(14, 4))
    ax = ax.flatten()

    for i in range(len(mass_titles)):

        mass_mask = (gal_mass > mass_bins[i]) & (gal_mass < mass_bins[i+1])
       
        median = np.nanmedian(np.log10(profiles[mass_mask* sf_mask]), axis=0)
        per25 = np.nanpercentile(np.log10(profiles[mass_mask* sf_mask]), 25, axis=0)
        per75 = np.nanpercentile(np.log10(profiles[mass_mask* sf_mask]), 75, axis=0)
        ax[i].plot(plot_bins, median, c=cb_blue, label='SF')
        #if i == 0:
        ax[i].fill_between(plot_bins, per25, per75, alpha=0.3, color=cb_blue)

        median = np.nanmedian(np.log10(profiles[mass_mask* gv_mask]), axis=0)
        per25 = np.nanpercentile(np.log10(profiles[mass_mask* gv_mask]), 25, axis=0)
        per75 = np.nanpercentile(np.log10(profiles[mass_mask* gv_mask]), 75, axis=0)
        ax[i].plot(plot_bins, median, c=cb_green, label='GV')
        #if i == 0:
        ax[i].fill_between(plot_bins, per25, per75, alpha=0.3, color=cb_green)

        median = np.nanmedian(np.log10(profiles[mass_mask* q_mask]), axis=0)
        per25 = np.nanpercentile(np.log10(profiles[mass_mask* q_mask]), 25, axis=0)
        per75 = np.nanpercentile(np.log10(profiles[mass_mask* q_mask]), 75, axis=0)
        ax[i].plot(plot_bins, median, c=cb_red, label='Q')
        #if i == 0:
        ax[i].fill_between(plot_bins, per25, per75, alpha=0.3, color=cb_red)

        ax[i].set_title(mass_titles[i])
        ax[i].set_xlabel(r'$r / r_{200}$')
        if i == 0:
            ax[i].legend()
            ax[i].set_ylabel(r'${\rm log} (Z / Z_{\odot})$')
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig('/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/mg_profile.png')
    plt.show()
    plt.clf()
