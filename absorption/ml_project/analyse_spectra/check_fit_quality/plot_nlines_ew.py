# Plot EW as a function of number of absorption lines

import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

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

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    x = [0.73, 0.68, 0.71, 0.69, 0.69, 0.7]
    orients = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    logN_min = 11.
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    delta_chisq = 0.05
    min_chisq = -2.
    max_chisq = 2
    chisq_bins = np.arange(min_chisq, max_chisq+delta_chisq, delta_chisq)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    cmap = cm.get_cmap('viridis')
    cmap = truncate_colormap(cmap, 0.1, 0.9)
    norm = colors.BoundaryNorm(np.arange(0.125, 1.625, 0.25), cmap.N)

    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')
    cax = plt.axes([0.15, 0.96, 0.7, 0.03])

    i = 0
    j = 0

    for line in lines:

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_N = []
        all_ew = []
        all_chisq = []
        all_fr200 = []
        all_orient = []
        all_ids = []

        for k in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N.extend(hf[f'log_N_{fr200[k]}r200'][:])
                all_ew.extend(hf[f'ew_{fr200[k]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[k]}r200'][:])
                all_orient.extend(hf[f'orient_{fr200[k]}r200'][:])
                all_fr200.extend([fr200[k]] * len(hf[f'ids_{fr200[k]}r200'][:]))
                all_ids.extend(hf[f'ids_{fr200[k]}r200'][:])

        all_N = np.array(all_N)
        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        all_fr200 = np.array(all_fr200)
        all_orient = np.array(all_orient)
        all_ids = np.array(all_ids)

        mask = (all_N > N_min[lines.index(line)]) * (all_chisq < chisq_lim[lines.index(line)]) 
        all_ew = all_ew[mask]
        all_N = all_N[mask]
        all_orient = all_orient[mask]
        all_fr200 = all_fr200[mask]
        all_ids = all_ids[mask]
    
        plot_ew_los = []
        plot_nlines = []
        plot_fr200 = []

        for k in range(len(gal_ids)):
            
            gal_mask = all_ids == gal_ids[k]
            if len(all_ids[gal_mask]) > 0:

                use_ew = all_ew[gal_mask]
                use_N = all_N[gal_mask]
                use_orient = all_orient[gal_mask]
                use_fr200 = all_fr200[gal_mask]
               
                # separate the arrays of ew by the LOS they belong to 
            
                for f in range(len(fr200)):
                    for o in range(len(orients)):
                        mask = (use_fr200 == fr200[f]) & (use_orient == orients[o])

                        plot_ew_los.append(np.sum(use_ew[mask]))
                        plot_nlines.append(len(use_ew[mask]))
                        plot_fr200.append(fr200[f])

        plot_ew_los = np.array(plot_ew_los)
        plot_nlines = np.array(plot_nlines)
        plot_fr200 = np.array(plot_fr200)

        plot_order = np.arange(len(plot_nlines))
        np.random.shuffle(plot_order)

        mask = plot_nlines > 0
        plot_ew_los = plot_ew_los[plot_order][mask]
        plot_nlines = plot_nlines[plot_order][mask]
        plot_fr200 = plot_fr200[plot_order][mask]

        im = ax[i][j].scatter(plot_nlines, np.log10(plot_ew_los),c=plot_fr200, 
                              cmap=cmap, norm=norm, s=8, alpha=0.6)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction',
                          bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log (EW}/\AA)$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'$n_{\rm abs}$')

        ax[i][j].set_xlim(0, 20)
        ax[i][j].set_ylim(-2, 1)

        if line == lines[3]:
            ax[i][j].set_yticks(np.arange(-2, 1., 0.5))
        if line in lines[3:5]:
            ax[i][j].set_xticks(np.arange(0, 20, 5))

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    fig.colorbar(im, cax=cax, ticks=fr200, label=r'$r_\perp / r_{200}$', orientation='horizontal')
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_nlines_ew_r200.png')
    plt.close()
