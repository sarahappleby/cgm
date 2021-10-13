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
cmap = truncate_colormap(cmap, 0.1, 0.9)

sf_cmap = plt.get_cmap('jet_r')
sf_cmap = truncate_colormap(sf_cmap, 0.1, 0.9)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_fgas = np.log10(sf['fgas'][:] + 1e-3)
        gal_Lbaryon = sf['L_baryon'][:]

    z = np.array([0, 0, 1])
    Lbaryon_norm = gal_Lbaryon / np.linalg.norm(gal_Lbaryon, axis=1)[:, None]
    alpha = np.degrees(np.arccos([np.dot(z, i) for i in Lbaryon_norm]))

    # for face-on galaxies at 90 degrees and edge on galaxies at 0 degrees:
    #inclination = np.abs(alpha - 90)
    #for i in range(len(inclination)):
    #        if (inclination[i] > 90) & (inclination[i] < 180):
    #        inclination[i] = np.abs(inclination[i] - 180)

    # for face-on galaxies at 0 degrees and edge on galaxies at 90 degrees:
    inclination = alpha.copy()
    for i in range(len(inclination)):
        if (alpha[i] > 90) & (alpha[i] < 180):
            inclination[i] = 180 - alpha[i]

    delta_m = 0.25
    min_m = 10.
    nbins_m = 6
    mass_bins = np.arange(min_m, min_m+nbins_m*delta_m, delta_m)

    sm_line = np.arange(9.5, 12.5, 0.5)
    sf_line = sfms_line(sm_line)
    q_line = sfms_line(sm_line, b=-8.7)

    plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgray')
    plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgray')
    plt.text(11.55, 1.1, 'SF')
    plt.text(11.55, 0.3, 'GV')
    plt.text(11.55, -0.7, 'Q')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3), c=inclination, cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'$i\ (^\circ)$')
    plt.clim(90, 0)
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, 1.5)
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_inclination.png')
    plt.clf()

    plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgray')
    plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgray')
    plt.text(11.55, 1.1, 'SF')
    plt.text(11.55, 0.3, 'GV')
    plt.text(11.55, -0.7, 'Q')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3), c=np.cos(alpha), cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'${\rm cos}\ i$')
    plt.clim(1, -1)
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, 1.5)
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_cosi.png')
    plt.clf()

    im = plt.scatter(gal_sm, inclination, c=np.log10(gal_sfr + 1e-3), cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.clim(1.5, -3.5)
    plt.xlim(9.75,11.75)
    plt.ylim(0, 90)
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'$i\ (^\circ)$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_mstar_inclination.png')
    plt.clf()

    im = plt.scatter(gal_sm, np.cos(alpha), c=np.log10(gal_sfr + 1e-3), cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.clim(1.5, -3.5)
    plt.xlim(9.75,11.75)
    plt.ylim(-1, 1)
    plt.xlabel(r'$\log\ (M_{\star} / M_{\odot})$')
    plt.ylabel(r'${\rm cos}\ i$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_mstar_cosi.png')
    plt.clf()

    plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgray')
    plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgray')
    plt.text(11.55, 1.1, 'SF')
    plt.text(11.55, 0.3, 'GV')
    plt.text(11.55, -0.7, 'Q')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, np.log10(gal_sfr + 1e-3), c=gal_fgas, cmap=cmap, s=5, marker='o')
    plt.colorbar(im, label=r'$\textrm{log} (f_{\textrm{gas}})$')
    plt.clim(1.0, -3.)
    plt.xlim(9.75,11.75)
    plt.ylim(-3.5, 1.5)
    plt.xlabel(r'$\log\ (M_{*} / M_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm SFR} / M_{\odot}{\rm yr}^{-1})$')
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_fgas.png')
    plt.clf()

