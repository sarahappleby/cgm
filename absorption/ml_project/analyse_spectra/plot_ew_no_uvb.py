import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict
sys.path.insert(0, '/disk04/sapple/tools/')
from plotmedian import runningmedian

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    line = 'H1215'
    plot_line = r'${\rm HI}1215$'
    norients = 8
    fr200 = 0.25
    minT = ['4.0', '4.5', '5.0', '5.5']
   
    delta_m = 0.25
    min_m = 10.
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    plot_bins = get_bin_middle(mass_bins)

    ssfr_cmap = plt.get_cmap('jet_r')
    ssfr_cmap = truncate_colormap(ssfr_cmap, 0.1, 1.0)
    colors = make_color_list(plt.get_cmap('viridis'), nbins_m)

    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/collisional/results/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_ssfr = sf['ssfr'][:] + 9.

    mass_long = np.repeat(gal_sm, norients)
    sfr_long = np.repeat(np.log10(gal_sfr + 1e-3), norients)
    ssfr_long = np.repeat(gal_ssfr, norients)

    # Median difference plots :)

    for i in range(len(minT)):

        no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_minT_{minT[i]}_ew_{line}.h5')
        with_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_with_uvb_minT_{minT[i]}_ew_{line}.h5')

        ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200}r200'].flatten()
        ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200}r200'].flatten()

        median = np.zeros(nbins_m)
        per25 = np.zeros(nbins_m)
        per75 = np.zeros(nbins_m)

        for j in range(nbins_m):

            mask = (mass_long > mass_bins[j]) & (mass_long < mass_bins[j+1])
            median[j] = np.nanmedian(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]))
            per25[j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 25.)
            per75[j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 75.)

        plt.plot(plot_bins, median, ls='-', c=colors[i], label=r'$T_{{\rm min}} = {{{}}}$'.format(minT[i]))
        if i == len(minT) -2:
            plt.fill_between(plot_bins, per25, per75, color=colors[i], alpha=0.4)
    
    plt.legend()
    plt.xlabel(r'$\log\ (M_{*} / M_{\odot})$')
    plt.ylabel(r'$\Delta {\rm log (EW}/\AA)$')
    plt.ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_mass_median_delta_ew_{line}.png')
    plt.show()
    plt.clf()


    """
    # Difference plots, coloured by sSFR :)

    fig, ax = plt.subplots(1, len(minT), figsize=(10, 6), sharey='row', sharex='col')
    ax = ax.flatten()

    for i in range(len(minT)):

        no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_minT_{minT[i]}_ew_{line}.h5')
        with_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_with_uvb_minT_{minT[i]}_ew_{line}.h5')

        ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200}r200'].flatten()
        ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200}r200'].flatten()

        im = ax[i].scatter(mass_long, np.log10(ew_no_uvb) - np.log10(ew_with_uvb), c=ssfr_long, cmap=ssfr_cmap, s=1.5, vmin=-2.5, vmax=0)
        ax[i].set_xlabel(r'$\log\ (M_{*} / M_{\odot})$')
        if i == 0:
            ax[i].set_ylabel(r'$\Delta {\rm log (EW}_{\rm HI}/\AA)$')
        if i == len(minT) -1:
            ax[i].annotate(plot_line, xy=(0.05, 0.85), xycoords='axes fraction')
        ax[i].set_title(r'$T_{{\rm min}} = {{{}}}$'.format(minT[i]))
        ax[i].set_ylim(-1, 1)

    fig.subplots_adjust(right=0.8, bottom=0.1, top=0.9)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im,ax=cbar_ax, shrink=.6, label=r'$\log\ ({\rm sSFR} / {\rm Gyr}^{-1})$')

    #plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_mass_delta_ew_{line}.png')
    plt.clf()
    """

