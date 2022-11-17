# Plot the EW from directly summing spectra vs the EW from the voigt fitting.

import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    fr200_choose = 0.25
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    orients = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    norients = 8
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    sample_file = f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5'
    with h5py.File(sample_file, 'r') as sf:
        gal_ids = sf['gal_ids'][:]


    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')
    cax = plt.axes([0.15, 0.96, 0.7, 0.03])

    i = 0
    j = 0

    for line in lines:

        sum_ew_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)

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
        all_chisq = all_chisq[mask]

        # for all impact parameters:
        sum_ew_data = np.zeros((len(gal_ids), len(fr200), len(orients)))
        fit_ew_data = np.zeros((len(gal_ids), len(fr200), len(orients)))
        fit_N_data = np.zeros((len(gal_ids), len(fr200), len(orients))) 
        chisq_data = np.ones((len(gal_ids), len(fr200), len(orients))) * 1e2

        for k in range(len(fr200)):
            sum_ew_data[:, k, :] = sum_ew_dict[f'ew_wave_{fr200[k]}r200']
            
        for k in range(len(gal_ids)):
            gal_mask = all_ids == gal_ids[k]
            
            for f in range(len(fr200)):
                fr200_mask = all_fr200 == fr200[f]
                
                for o in range(len(orients)):
                    orient_mask = all_orient == orients[o]
                    mask = gal_mask * fr200_mask * orient_mask
                    fit_ew_data[k][f][o] = np.nansum(all_ew[mask])
                    fit_N_data[k][f][o] = np.nansum(10**all_N[mask])
                    if len(all_chisq[mask]) > 0:
                        chisq_data[k][f][o] = np.max(all_chisq[mask])

        data_x = np.log10(sum_ew_data + 10**-2.).flatten()
        data_y = np.log10(fit_ew_data + 10**-2.).flatten()
        data_c = np.log10(chisq_data).flatten()
        
        im = ax[i][j].scatter(data_x, data_y, c=data_c, s=1, cmap='magma', vmin=-1)
        ax[i][j].plot(np.arange(-4, 4), np.arange(-4, 4), c='k', ls='--', lw=1)
       
        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log (EW}/{\rm \AA})_{\rm fit}$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log (EW}/{\rm \AA})_{\rm sum}$')

        ax[i][j].set_xlim(-2.25, 1.25)
        ax[i][j].set_ylim(-2.25, 1.25)
        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.92), xycoords='axes fraction')
       
        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    fig.colorbar(im, cax=cax, label=r'${\rm log}\ \chi^2_r$', orientation='horizontal')
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_compare_ew_fit_lines.png')
    plt.clf()


