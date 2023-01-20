# Playing with definitions of star forming, green valley, quenched

import caesar
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rcParams["figure.figsize"] = (7,6)

def belfiore_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sfms_line(mstar, a=1., b=-10.8):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap


model = 'm100n1024'
wind = 's50'
snap = '151'

sf_cmap = plt.get_cmap('jet_r')
sf_cmap = truncate_colormap(sf_cmap, 0.1, 0.9)

delta_m = 0.25
min_m = 10.
nbins_m = 5
mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
ngals_each = 12
nbins_ssfr = 3

sfr_belfiore = sfms_line(11.0)

sample_dir = f'/disk04/sapple/data/samples/'
data_dir = f'/home/rad/data/{model}/{wind}/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
redshift = sim.simulation.redshift

gal_cent = np.array([i.central for i in sim.galaxies])
gal_sm = np.array([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))])
gal_sfr = np.array([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))])
gal_ssfr = np.log10((gal_sfr / gal_sm) + 1.e-14) 
gal_sm = np.log10(gal_sm)
gal_sfr = np.log10(gal_sfr + 1.e-3)
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

sm_line = np.arange(9.5, 12.5, 0.5)
ssfr_b = ssfr_b_redshift(redshift)
belfiore_sf_line = belfiore_line(sm_line, b=ssfr_b)
belfiore_q_line = belfiore_line(sm_line, b=ssfr_b-1.)

pivot_sf_line = sfms_line(sm_line, a=1., b=-10.67)
pivot_q_line = sfms_line(sm_line, a=1, b=-11.67)
thresh_sf_line = sfms_line(sm_line, a=1, b=-10.8)
thresh_q_line = sfms_line(sm_line, a=1, b=-11.8)

im = plt.scatter(gal_sm, gal_sfr, c=gal_ssfr +9., cmap=sf_cmap, s=5, marker='o')
plt.colorbar(im, label=r'$\textrm{log} ({\rm sSFR} / {\rm Gyr}^{-1})$')
plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
plt.clim(-3.0, 0.5)
plt.xlim(9.75,11.75)
plt.plot(sm_line, belfiore_sf_line, ls='-', lw=1.5, c='cornflowerblue', label='Belfiore')
plt.plot(sm_line, belfiore_q_line, ls='-', lw=1.5, c='cornflowerblue')
plt.plot(sm_line, pivot_sf_line, ls='--', lw=1.5, c='royalblue', label='Belfiore pivoted')
plt.plot(sm_line, pivot_q_line, ls='--', lw=1.5, c='royalblue')
plt.plot(sm_line, thresh_sf_line, ls='--', lw=1.5, c='navy', label=r'${\rm sSFR} > -1.8$')
plt.plot(sm_line, thresh_q_line, ls='--', lw=1.5, c='navy')
plt.legend()
for i in range(nbins_m + 1):
    plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
plt.savefig('m100n1024_s50_151_sf_defintion.png')
plt.clf()
