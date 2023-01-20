import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.1, 1.0)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    redshift = 0.
    chisq_lim = 2.5
    N_min = 12.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:] + 9.

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        fig, ax = plt.subplots(1, len(fr200), figsize=(14, 5), sharey='row', sharex='col')
        ax = ax.flatten()

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_b = hf[f'b_{fr200[i]}r200'][:]
                all_l = hf[f'l_{fr200[i]}r200'][:]
                #all_ew = hf[f'ew_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_chisq < chisq_lim)
            all_N = all_N[mask]
            all_b = all_b[mask]
            all_l = all_l[mask]
            #all_ew = all_ew[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten() 
            all_mass = mass[idx]
            all_ssfr = ssfr[idx]
            
            im = ax[i].scatter(all_mass, all_N, c=all_ssfr, cmap=cmap, s=1, vmin=-2, vmax=0)
            if i == len(fr200) -1:
                fig.colorbar(im, ax=ax[i], label=r'$\textrm{log} ({\rm sSFR} / {\rm Gyr}^{-1})$')
            ax[i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            ax[i].set_xlim(10, 12)
            ax[i].set_ylim(12, 18)
            ax[i].set_xlabel(r'${\rm log }(M_\star / {\rm M}_{\odot})$')
            if i == 0:
                ax[i].set_ylabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_mass_column_{line}.png')
        plt.clf()


