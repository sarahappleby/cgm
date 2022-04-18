import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.1, 1.0)

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$', 
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 2.5
    
    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_ssfr = sf['ssfr'][:] + 9.

    mass_long = np.repeat(gal_sm, norients)
    sfr_long = np.repeat(np.log10(gal_sfr + 1e-3), norients)
    ssfr_long = np.repeat(gal_ssfr, norients)

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):
        fitN_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_ew_{line}.h5')

        for i in range(len(fr200)):
            ew = fitN_dict[f'ew_wave_{fr200[i]}r200'].flatten()

            im = ax[l][i].scatter(mass_long, np.log10(ew), c=ssfr_long, cmap=cmap, s=1.5, vmin=-2.5, vmax=0)
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'$\log\ (M_{*} / M_{\odot})$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log (EW}/\AA)$')
            if i == len(fr200) -1:
                ax[l][i].annotate(plot_lines[l], xy=(0.05, 0.85), xycoords='axes fraction')
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[l][i].set_ylim(-2, 1)
   
    fig.subplots_adjust(right=0.8, bottom=0.1, top=0.9)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im,ax=cbar_ax, shrink=.6, label=r'$\log\ ({\rm sSFR} / {\rm Gyr}^{-1})$')        

    #plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_mass_ew.png')
    plt.show()
    plt.clf()

