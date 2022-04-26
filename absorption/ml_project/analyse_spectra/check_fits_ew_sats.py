import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    norients = 8
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

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]


    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:
        
        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/satellites/results/{model}_{wind}_{snap}_log_frad_1_fit_lines_{line}.h5'

        ax[i][j].axhline(0.9, c='k', ls=':', lw=1)

        all_ew = []
        all_chisq = []

        for k in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_ew.extend(hf[f'ew_{fr200[k]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[k]}r200'][:])

        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        print(line, np.nanmedian(all_chisq))

        mask = (np.log10(all_chisq) < max_chisq)
        all_ew = all_ew[mask]
        all_chisq = all_chisq[mask]

        print(line, np.nanmedian(all_chisq))

        ew_total = np.nansum(all_ew)
        few_total = np.zeros(len(chisq_bins))

        for l in range(len(chisq_bins) -1):
            mask = (np.log10(all_chisq) < chisq_bins[l+1])
            few_total[l] = np.nansum(all_ew[mask])/ew_total
        few_total[-1] = few_total[-2]

        chisq_lim = chisq_bins[np.argmin(np.abs(few_total - 0.9))]
        ax[i][j].axvline(chisq_lim, c='k', ls=':', lw=1)
        print(line, 10**chisq_lim)

        ax[i][j].step(chisq_bins, few_total, c=colors[2], lw=1, ls='-')

        ax[i][j].set_xlim(min_chisq, max_chisq)
        ax[i][j].set_ylim(0, 1.05)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.9), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm EW} / {\rm EW}_{\rm Total}$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log}\ \chi^2_r$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

        print('\n')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sats_chisq_few_all_r200.png')
    plt.close()

    """
    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        ax[i][j].axhline(0.9, c='k', ls=':', lw=1)

        for k in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_ew = hf[f'ew_{fr200[k]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[k]}r200'][:]   
                        
            mask = (np.log10(all_chisq) < max_chisq)
            all_ew = all_ew[mask]
            all_chisq = all_chisq[mask]

            ew_total = np.nansum(all_ew)
            few_total = np.zeros(len(chisq_bins))
            
            for l in range(len(chisq_bins) -1):
                mask = (np.log10(all_chisq) < chisq_bins[l+1])
                few_total[l] = np.nansum(all_ew[mask])/ew_total
            few_total[-1] = few_total[-2]

            chisq_lim = chisq_bins[np.argmin(np.abs(few_total - 0.9))]
            ax[i][j].axvline(chisq_lim, c='k', ls=':', lw=1)

            ax[i][j].step(chisq_bins, few_total, c=colors[k], lw=1, ls='-', label=r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[k]))
        
        ax[i][j].set_xlim(min_chisq, max_chisq)
        ax[i][j].set_ylim(0, 1.05)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.9), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        if line == 'H1215':
            ax[i][j].legend(loc=4)
       
        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm EW} / {\rm EW}_{\rm Total}$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log}\ \chi^2_r$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0
 
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_chisq_few.png')
    plt.close()
    """
