import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import caesar
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

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0.1, 0.9)

if __name__ == '__main__':

    model = 'm100n1024'
    snap = sys.argv[1]
    wind = 's50'

    snaps = ['151', '137', '125', '105', '078']
    snap_index = snaps.index(snap)
    sf_height = [1.1, 1.2, 1.6, 2.2, 2.6]
    gv_height = [0.3, 0.3, 0.5, 0.7, 1.1]
    q_height = [-0.7, -0.7, -0.5, -0.3, 0.]

    datadir = f'/home/rad/data/{model}/{wind}/Groups/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    sim = caesar.load(f'{datadir}{model}_{snap}.hdf5')

    delta_m = 0.25
    min_m = 10.
    nbins_m = 6
    mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)

    sm_line = np.arange(9.5, 12.5, 0.5)
    # this comes from the b=-7.7 at z=0 from Belfiore+2018 and getting the star forming
    # main sequence at z=2 (b=-6.8), and linearly log(1+z) to the two end points
    ssfr_b = 1.9*np.log10(1+sim.simulation.redshift)-7.7
    sf_line = sfms_line(sm_line, b=ssfr_b)
    q_line = sfms_line(sm_line, b=ssfr_b - 1.)

    gal_cent = np.array([i.central for i in sim.galaxies])
    gal_sm = np.array([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))])
    gal_sfr = np.array([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))])
    gal_ssfr = gal_sfr / gal_sm
    gal_ssfr = np.log10(gal_ssfr + 1e-14)+9.
    gal_sm = np.log10(gal_sm)
    gal_sfr = np.log10(gal_sfr + 1.e-3)
    gal_fgas = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ]) 
    gal_fgas = np.log10(gal_fgas + 1.e-3)

    im = plt.scatter(gal_sm[gal_cent], gal_sfr[gal_cent], c=gal_fgas[gal_cent], cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'${\rm log}\ f_{\rm gas}$')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgrey')
    plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgrey')
    plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgrey')
    plt.text(11.55, sf_height[snap_index], r'${\bf SF}$', color='dimgrey')
    plt.text(11.55, gv_height[snap_index], r'${\bf GV}$', color='dimgrey')
    plt.text(11.55, q_height[snap_index], r'${\bf Q}$', color='dimgrey')
    plt.clim(-3., 1.)
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, 3.5)
    plt.xlabel(r'${\rm log}\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'${\rm log}\ ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_sfr.png')
    plt.clf()
