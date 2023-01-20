import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
from scipy.optimize import curve_fit
from scipy import interpolate
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def power_law(x, a, b):
    return x*a + b

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'

    ncells = 16
    start = [13.75, 12.75, 13.75, 12.75, 13.75, 13.75]
    end = [15.5, 14.5, 15.5, 14.5, 15.5, 14.5]
    logN = np.arange(9, 18, 0.01)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    i = 0
    j = 0

    for l, line in enumerate(lines):

        cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'
        #cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_mass_small.h5'
        #cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_mass.h5'
        #cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion_extras.h5'
        #cddf_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_mass_extras.h5'

        plot_data = read_h5_into_dict(cddf_file)

        ax[i][j].plot(plot_data['plot_logN'], plot_data[f'cddf_all'], c='dimgrey', ls='', lw=1)

        mask = ~np.isinf(plot_data['cddf_all'])
        logN_use = plot_data['plot_logN'][mask]
        data_use = plot_data['cddf_all'][mask]
        start_i = np.argmax(data_use)

        popt, pcov = curve_fit(power_law, logN_use[start_i:], data_use[start_i:])
        power_law_fit = logN*popt[0]+ popt[1]
        ax[i][j].plot(logN, power_law_fit, c='tab:pink', lw=1, ls='--')

        if line == 'CII1334':
            f = interpolate.interp1d(plot_data['plot_logN'][1:], plot_data['cddf_all'][1:], fill_value='extrapolate')
        elif line == 'OVI1031':
            mask = ~np.isinf(plot_data['cddf_all']) 
            f = interpolate.interp1d(plot_data['plot_logN'][mask], plot_data['cddf_all'][mask], fill_value='extrapolate')
        else:
            f = interpolate.interp1d(plot_data['plot_logN'], plot_data['cddf_all'], fill_value='extrapolate') 
        cddf_extra = f(logN)
        ax[i][j].plot(logN, cddf_extra, c='skyblue', lw=1, ls='--')

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log }(\delta^2 n / \delta X \delta N )$')

        if ('extras' in cddf_file) & (line == 'CIV1548'):
            mask = (logN >= 11.75)
        else:
            mask = np.ones(len(logN)).astype(bool)

        end_i = np.argmax(cddf_extra[mask]) + 10

        plot_data['completeness'] = logN[mask][np.argmin(np.abs((10**cddf_extra[mask][:end_i] / 10**power_law_fit[mask][:end_i] ) - 0.5))]
        print(plot_data['completeness'])
        ax[i][j].axvline(plot_data['completeness'], c='k', ls='--', lw=1)

        write_dict_to_h5(plot_data, cddf_file)

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_completeness.png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_cddf_completeness_extras.png')
    plt.show()
    plt.close()
