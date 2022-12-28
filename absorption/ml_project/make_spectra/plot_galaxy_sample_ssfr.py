# Plot our galaxy sample

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import numpy as np
import sys
import caesar

def ssfr_b_redshift(z):
    return 1.9*np.log10(1+z) - 7.7

def belfiore_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b

def sfms_line(mstar, a=1., b=-10.8):
    return mstar*a + b

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def runningmedian(x,y,xlolim=-1.e20,ylolim=-1.e20,bins=10,stat='median'):
    
    xp = x[(x>xlolim) & (y>ylolim)]
    yp = y[(x>xlolim) & (y>ylolim)]
    hist, bin_edges = np.histogram(xp, bins)
    bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ymedian = np.zeros(bins)
    ymean = np.zeros(bins)
    ysigma = np.zeros(bins)
    yper25 = np.zeros(bins)
    yper75 = np.zeros(bins)
    ndata = np.zeros(bins)
    
    for i in range(bins):
        
        mask = (xp > bin_edges[i]) & (xp < bin_edges[i+1])
        
        ymedian[i] = np.median(10**yp[mask])
        ymean[i] = np.mean(10**yp[mask])
        ysigma[i] = np.std(10**yp[mask])
        yper25[i] = np.percentile(10**yp[mask], 25.)
        yper75[i] = np.percentile(10**yp[mask], 75.)
        ndata[i] = len(yp[mask])

    ymean, ysigma = convert_to_log(ymean, ysigma)
    yper_lo = ymedian - yper25; yper_hi = yper75 - ymedian
    ymedian, ypers = convert_to_log(ymedian, [yper_lo, yper_hi])
    return bin_cent,ymean,ysigma,ymedian, ypers,ndata


cmap = plt.get_cmap('plasma')
cmap = truncate_colormap(cmap, 0.1, 0.9)

greys = plt.get_cmap('Greys')
greys = truncate_colormap(greys, 0.0, 0.5)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    mass_min = 9.75
    mass_max = 11.5

    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim = caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5') 
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)+9

    gwslc_data_file = '/home/rad/gizmo-analysis/caesar/Observations/GSWLC/GSWLC-X2.dat'
    gwslc_data = {}
    gwslc_data['mass'], gwslc_data['sigma_mass'], gwslc_data['sfr'], gwslc_data['sigma_sfr'], gwslc_data['flag_sed'], gwslc_data['flag_mgs'] = \
            np.loadtxt(gwslc_data_file,usecols=(9,10,11,12,19,23),unpack=True)
    gwslc_data['ssfr'] = gwslc_data['sfr'] - gwslc_data['mass'] +9
    flag_mask = (gwslc_data['flag_sed']==0) & (gwslc_data['flag_mgs']==1) & (gwslc_data['ssfr'] > quench)
    type_mask = (gwslc_data['mass'] > mass_min) & (gwslc_data['mass'] < mass_max)
    mask = flag_mask * type_mask
    bin_cent, _, ysigma, ymedian, _, ndata = runningmedian(gwslc_data['mass'][mask],gwslc_data['ssfr'][mask],xlolim=-1.e20,ylolim=-1.e20,bins=10,stat='median')
    plt.errorbar(bin_cent, ymedian, yerr=ysigma, c='#6e6b6a', ls='', marker='s', lw=1, markersize=5, capsize=2, label='GSWLC')

    possible_snaps = ['151', '137', '125', '105', '078']
    snap_index = possible_snaps.index(snap)
    sf_height = [1.1, 1.2, 1.6, 2.1, 2.6]
    gv_height = [0.3, 0.3, 0.5, 0.7, 1.1]
    q_height = [-0.7, -0.7, -0.5, -0.3, 0.]
    q_height = [-3.05]*5
    ylims = [1.5, 1.75, 2., 2.5, 3.0]

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    with h5py.File(sample_file, 'r') as sf:
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_ssfr = sf['ssfr'][:]+9
        gal_fgas = np.log10(sf['fgas'][:] + 1e-3)
        gal_Lbaryon = sf['L_baryon'][:]
        #gal_nsats = sf['nsats'][:]
        gal_Tcgm = sf['Tcgm'][:]
        gal_fcold = sf['fcold'][:]
    gal_ssfr[gal_ssfr == -5] +=1

    gal_sm_ssfr_file = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}_sm_ssfr.h5'
    with h5py.File(gal_sm_ssfr_file, 'r') as hf:
        gal_sm_ssfr_hist2d = hf['sm_ssfr'][:]
        mass_bins = hf['mass_bins'][:]
        ssfr_bins = hf['ssfr_bins'][:]

    delta_m_hist = mass_bins[1] - mass_bins[0]
    delta_ssfr_hist = ssfr_bins[1] - ssfr_bins[0]
    aspect = (delta_m_hist / delta_ssfr_hist) * 0.9

    delta_m = 0.25
    min_m = 10.
    nbins_m = 6

    plt.imshow(np.log10(gal_sm_ssfr_hist2d), extent=(mass_bins[0], mass_bins[-1], ssfr_bins[0], ssfr_bins[-1]),aspect=aspect, cmap=greys, label='Simba')

    #plt.plot(sm_line, sf_line, ls='--', lw=1.3, c='dimgray')
    #plt.plot(sm_line, q_line, ls='--', lw=1.3, c='dimgray')
    plt.axhline(quench, ls='--', lw=1.3, c='dimgray')
    plt.axhline(quench-1, ls='--', lw=1.3, c='dimgray')
    plt.text(11.56, quench+0.5, 'SF')
    plt.text(11.55, quench-0.6, 'GV')
    plt.text(11.575, quench-1.6, 'Q')
    for i in range(nbins_m + 1):
        plt.axvline(min_m+i*delta_m, ls=':', lw=1.5, c='darkgray')
    im = plt.scatter(gal_sm, gal_ssfr, c=gal_Tcgm, cmap=cmap, s=10, marker='o', vmin=5, vmax=7, label='Simba sample')
    plt.colorbar(im, label=r'${\rm log} (T_{\rm CGM} / {\rm K})$')
    plt.xlim(9.75,11.75)
    plt.ylim(-4.1, 0)
    plt.xlabel(r'$\log\ (M_{\star} / {\rm M}_{\odot})$')
    plt.ylabel(r'$\textrm{log} ({\rm sSFR} / {\rm Gyr}^{-1})$')
    plt.legend(loc=1, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{sample_dir}{model}_{wind}_{snap}_Tcgm_ssfr.pdf', format='pdf')
    plt.close()
