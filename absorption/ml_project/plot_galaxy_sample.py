# Plot our galaxy sample

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import numpy as np
import sys

def sfms_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0., 0.95)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sample_dir = f'/disk01/sapple/cgm/absorption/ml_project/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]

    delta_m = 0.25
    min_m = 10.
    nbins_m = 6
    mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)

    sm_line = np.arange(9.5, 12., 0.5)
    sf_line = sfms_line(sm_line)
    q_line = sfms_line(sm_line, b=-8.7)

    plt.plot(sm_line, sf_line, ls='--', lw=1.8, c='k')
    plt.plot(sm_line, q_line, ls='--', lw=1.8, c='k')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.8, c='k')
    plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3))
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, 2.)
    plt.xlabel(r'$\log\ (M_{*} / M_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_nocmap.png')
    plt.clf()

