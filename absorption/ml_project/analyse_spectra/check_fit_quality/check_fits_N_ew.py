# Plot the EW of the fit against N, and get an expression for fitted EW against N. Use to estimate how much absorption
# is missing from the fitted sample.

import numpy as np
import h5py
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra')
from utils import read_h5_into_dict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def linear(x, a, b):
    return a*x + b

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    orients = [0, 45, 90, 135, 180, 225, 270, 315]
    norients = len(orients)
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    logN = np.arange(9, 18, 0.01)
    logew = np.arange(-3, 1, 0.01)
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    width = 0.015
    height = 0.77
    vertical_position = 0.11
    horizontal_position = 0.9

    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]

    fits_a = np.zeros(len(lines))
    fits_b = np.zeros(len(lines))

    ### N line against EW line
    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_N = []
        all_ew = []
        all_chisq = []

        for k in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_N.extend(hf[f'log_N_{fr200[k]}r200'][:])
                all_ew.extend(hf[f'ew_{fr200[k]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[k]}r200'][:])

        all_N = np.array(all_N)
        all_ew = np.array(all_ew)
        all_chisq = np.array(all_chisq)
        all_ew[all_ew < 0] = 0

        data_x = np.log10(all_ew + 1.e-3)
        data_y = all_N
        data_c = np.log10(all_chisq)
        data_c[np.isnan(data_c)] = 3

        im = ax[i][j].scatter(data_x, data_y, c=data_c, s=1)
    
        mask = (data_y > 10) & (data_y < 15)

        popt, pcov = curve_fit(linear, data_x[mask], data_y[mask])
        fits_a[lines.index(line)] = popt[0]
        fits_b[lines.index(line)] = popt[1]
        linear_fit = logew*popt[0]+ popt[1]
        ax[i][j].plot(logew, linear_fit, c='tab:pink', lw=1, ls='--')

        ax[i][j].set_xlim(-3, 1)
        ax[i][j].set_ylim(11, 19)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.9), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log }(N / {\rm cm}^{-2})$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log (EW}/\AA)$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    ax[0][0].set_yticks(np.arange(12, 20, 1))
    ax[1][0].set_xticks(np.arange(-3, 1))
    ax[1][1].set_xticks(np.arange(-3, 1))

    cax = plt.axes([horizontal_position, vertical_position, width, height])
    fig.colorbar(im, cax=cax, label=r'${\rm log}\ \chi^2_r$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fits_ew_N.png')
    plt.close()

    ### Predicted N against total EW
    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        sum_ew_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)
        all_sum_ew = np.zeros(( len(fr200), len(sum_ew_dict[f'ew_wave_0.25r200']) * norients))

        for k in range(len(fr200)):
            all_sum_ew[k] = sum_ew_dict[f'ew_wave_{fr200[k]}r200'].flatten()
     
        all_sum_ew[all_sum_ew < 0] = 0
        all_sum_ew = np.log10(all_sum_ew.flatten() + 1e-3)
        predict_N = all_sum_ew * fits_a[lines.index(line)] + fits_b[lines.index(line)]

        ax[i][j].hexbin(all_sum_ew, predict_N, bins='log', cmap='Blues')

        ax[i][j].set_xlim(-3.1, 1)
        ax[i][j].set_ylim(11, 19)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.9), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log (EW}_{\rm total}/\AA)$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }(N_{\rm expected} / {\rm cm}^{-2})$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fits_total_ew_expected_N.png')
    plt.close()


    ### Predicted N against line N
    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        sum_ew_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)
        
        results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_total_ew = []
        all_N_predict = []

        for k in range(len(fr200)):

            sum_ew = sum_ew_dict[f'ew_wave_{fr200[k]}r200']
            predict_N = sum_ew * fits_a[lines.index(line)] + fits_b[lines.index(line)]

            with h5py.File(results_file, 'r') as hf:
                all_N = hf[f'log_N_{fr200[k]}r200'][:]
                all_chisq = hf[f'chisq_{fr200[k]}r200'][:]
                all_ids = hf[f'ids_{fr200[k]}r200'][:]
                all_orients = hf[f'orient_{fr200[k]}r200'][:]

        idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()        
        orient_idx = np.array([np.where(orients == j)[0] for j in all_orients]).flatten()

        data_x = []
        data_y = []
        data_c = []
        
        for k in range(len(all_N)):
            data_x.append(predict_N[idx[k]][orient_idx[k]])
            data_y.append(all_N[k])
            data_c.append(all_chisq[k])

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_c = np.log10(all_chisq)
        data_c[np.isnan(data_c)] = 3

        im = ax[i][j].scatter(data_x, data_y, c=data_c, s=1)

        ax[i][j].set_xlim(14, 17)
        ax[i][j].set_ylim(11, 18)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(0.05, 0.9), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", lw=0.75))

        ax[i][j].plot(np.arange(10, 20), np.arange(10, 20), ls=':', c='k', lw=1)

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log }(N_{\rm individual} / {\rm cm}^{-2})$')
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }(N_{\rm expected} / {\rm cm}^{-2})$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    ax[0][0].set_yticks(np.arange(12, 19, 1))
    ax[1][0].set_xticks(np.arange(14, 17))
    ax[1][1].set_xticks(np.arange(14, 17))

    cax = plt.axes([horizontal_position, vertical_position, width, height])
    fig.colorbar(im, cax=cax, label=r'${\rm log}\ \chi^2_r$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fits_N_expected_N.png')
    plt.close()

