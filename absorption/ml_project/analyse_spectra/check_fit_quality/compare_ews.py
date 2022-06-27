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


def make_comparison_plots(fit_data, sum_data, chisq_data, plot_name):

    im = plt.scatter(data_x, data_y, c=data_c, s=1, cmap='magma')
    plt.colorbar(im, label=r'${\rm log}\ \chi^2_r$')
    plt.clim(-3, 3)
    plt.plot(np.arange(-4, 4), np.arange(-4, 4), c='k', ls='--', lw=1)
    plt.ylabel(r'${\rm log (EW}/\AA)_{\rm fit}$')
    plt.xlabel(r'${\rm log (EW}/\AA)_{\rm sum}$')
    plt.xlim(-3.25, 2)
    plt.ylim(-3.25, 2)
    plt.annotate(plot_lines[i], xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(f'{plot_name}_compare_ew_fit.png')
    plt.clf()

    fig, ax = plt.subplots(1, 1)
    hb = ax.hexbin(data_x[~np.isnan(data_x) * ~np.isnan(data_y)], data_y[~np.isnan(data_x) * ~np.isnan(data_y)], gridsize=50, bins='log', cmap='Blues')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r'${\rm log }N$')
    plt.plot(np.arange(-4, 4), np.arange(-4, 4), c='k', ls='--', lw=1)
    plt.ylabel(r'${\rm log (EW}/\AA)_{\rm fit}$')
    plt.xlabel(r'${\rm log (EW}/\AA)_{\rm sum}$')
    plt.xlim(-3.25, 2)
    plt.ylim(-3.25, 2)
    plt.annotate(plot_lines[i], xy=(0.05, 0.95), xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(f'{plot_name}_compare_ew_fit_hex.png') 
    plt.clf()


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    fr200_choose = 0.25
    #chisq_lim = [4.5, 63.1, 20.0, 70.8, 15.8, 4.5] limits with old fitting procedure
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    norients = 8
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    for i, line in enumerate(lines):

        chisq_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_chisq_{line}.h5'
        chisq_dict = read_h5_into_dict(chisq_file)
        fit_ew_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_ew_{line}.h5'
        fit_ew_dict = read_h5_into_dict(fit_ew_file)
        sum_ew_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)

        """
        # for 0.25 r200 only:
        data_x = np.log10(sum_ew_dict[f'ew_wave_{fr200_choose}r200']+1e-3).flatten()
        data_y = np.log10(fit_ew_dict[f'fit_ew_{fr200_choose}r200'] + 1e-3).flatten()
        data_c = np.log10(chisq_dict[f'max_chisq_{fr200_choose}r200']).flatten()
        data_c[np.isnan(data_c)] = 3.

        make_comparison_plots(data_x, data_y, data_c, f'{plot_dir}{model}_{wind}_{snap}_{line}_{fr200_choose}r200')
        """

        # for all impact parameters:
        all_fit_data = np.zeros(( len(fr200), len(sum_ew_dict[f'ew_wave_{fr200_choose}r200']) * norients))
        all_sum_data = np.zeros(( len(fr200), len(sum_ew_dict[f'ew_wave_{fr200_choose}r200']) * norients))
        all_chisq_data = np.zeros(( len(fr200), len(sum_ew_dict[f'ew_wave_{fr200_choose}r200']) * norients))

        for j in range(len(fr200)):
            all_fit_data[j] = fit_ew_dict[f'fit_ew_{fr200[j]}r200'].flatten()
            all_sum_data[j] = sum_ew_dict[f'ew_wave_{fr200[j]}r200'].flatten()
            all_chisq_data[j] = chisq_dict[f'max_chisq_{fr200[j]}r200'].flatten()

        data_x = np.log10(all_sum_data + 1.e-3).flatten()
        data_y = np.log10(all_fit_data + 1.e-3).flatten()
        data_c = np.log10(all_chisq_data).flatten()
        data_c[np.isnan(data_c)] = 3

        make_comparison_plots(data_x, data_y, data_c, f'{plot_dir}{model}_{wind}_{snap}_{line}')
