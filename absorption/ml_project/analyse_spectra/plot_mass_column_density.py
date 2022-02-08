import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.05, 1.0)

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$', 
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['sf_med', 'sf_per25', 'sf_per75', 'gv_med', 'gv_per25', 'gv_per75', 'q_med', 'q_per25', 'q_per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 2.5

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]
        gal_ssfr = sf['ssfr'][:] + 9.

    mass_long = np.repeat(gal_sm, norients)
    sfr_long = np.repeat(np.log10(gal_sfr + 1e-3), norients)
    ssfr_long = np.repeat(gal_ssfr, norients)
    sf_mask, gv_mask, q_mask = ssfr_type_check(redshift, mass_long, sfr_long)

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'

    fig, ax = plt.subplots(1, , sharey='row')
    ax = ax.flatten()

    for l, line in enumerate(lines):
        chisq_dict = read_h5_into_dict(f'{results_dir}fit_max_chisq_{line}.h5')    
        mask_dict = {}
        for key in chisq_dict.keys():
            mask_dict[f'{key}_mask'] = np.abs(chisq_dict[key]) < chisq_lim
        del chisq_dict 
        fitN_dict = read_h5_into_dict(f'{results_dir}fit_column_densities_{line}.h5')

        fig, ax = plt.subplots(1, 5, sharey='row')
        ax = ax.flatten()

        plot_data = {}
        for pq in plot_quantities:
            plot_data[pq] = np.zeros(len(fr200))

        for i in range(len(fr200)):
            totalN = fitN_dict[f'log_totalN_{fr200[i]}r200'].flatten()
            chisq_mask = mask_dict[f'max_chisq_{fr200[i]}r200_mask'].flatten()

            im = ax[i].scatter(mass_long[mask], totalN[mask], c=ssfr_long[mask], cmap=cmap, s=1, vmin=-2, vmax=0)
            ax[i].set_xlabel(r'$\log\ (M_{*} / M_{\odot})$')
            ax[i].set_ylabel(r'$\log\ N ({\rm cm}^{-2})$')
            cbar = fig.colorbar(im,ax=ax, label=r'$\log\ ({\rm sSFR} / {\rm Gyr}^{-1})$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_total_column_profile.png')
    plt.clf()

