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
    snap = '125'

    cmap = plt.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.1, 1.0)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    redshift = 0.
    chisq_lim = 2.5
    N_min = 12.
    N_max = 18.
    b_max = 10**2.5

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

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[i]}r200'][:]
                all_b = hf[f'b_{fr200[i]}r200'][:]
                all_l = hf[f'l_{fr200[i]}r200'][:]
                #all_ew = hf[f'ew_{fr200[i]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[i]}r200'][:]
                all_ids = hf[f'ids_{fr200[i]}r200'][:]

            mask = (all_N > N_min) * (all_N < N_max) * (all_b < b_max) * (all_chisq < chisq_lim)
            all_N = all_N[mask]
            all_b = all_b[mask]
            all_l = all_l[mask]
            #all_ew = all_ew[mask]

            all_ids = all_ids[mask]
            idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten() 
            all_ssfr = ssfr[idx]
            
            
            hb = ax[l][i].hexbin(all_N, np.log10(all_b), gridsize=50, bins='log', cmap='Blues')
    
            #im = ax[l][i].scatter(all_N, np.log10(all_b), c=all_ssfr, cmap=cmap, s=1, vmin=-2, vmax=0)
            ax[l][i].set_xlim(12, 18) 
            ax[l][i].set_ylim(0, 2.5)
            
            if i == len(fr200) -1:
                fig.colorbar(hb, ax=ax[l][i], label=r'${\rm log} n$')
            if l == 0:
                ax[l][i].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
            if l == len(lines)-1:
                ax[l][i].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')
            if i == 0:
                ax[l][i].set_ylabel(r'${\rm log }(b /{\rm km s}^{-1})$')
                ax[l][i].annotate(plot_lines[l], xy=(0.65, 0.85), xycoords='axes fraction')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_kinematics_hexbin.png')
    plt.show()
    plt.clf()


