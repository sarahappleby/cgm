import matplotlib.pyplot as plt
import h5py
import numpy as np
import pygad as pg
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from spectrum import Spectrum
from utils import *
from physics import *


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    results_dir = f'/disk04/sapple/data/normal/results/'
    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(12, 14), sharey='row')

    for i in range(len(lines)):

        results_file = f'/disk04/sapple/data/satellites/results/{model}_{wind}_{snap}_log_frad_1_fit_lines_{lines[i]}.h5'
        
        for j in range(nbins_fr200):

            with h5py.File(results_file, 'r') as hf:
                all_chisq = hf[f'chisq_{fr200[j]}r200'][:]

            mask = np.abs(np.log10(all_chisq)) < 2.

            ax[i][j].hist(np.log10(all_chisq[mask]), bins=100, density=True, alpha=0.6)
            ax[i][j].axvline(np.log10(2.5), ls='--', c='k')
            if i == len(lines) -1:
                ax[i][j].set_xlabel(r'${\rm log}\ \chi^2_r$')
            if j == 0:
                ax[i][j].set_ylabel(r'${\rm Frequency}$')
                ax[i][j].annotate(plot_lines[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=13)
            if i == 0:
                ax[i][j].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[j]))
            ax[i][j].set_xlim(-2, 2)
            ax[i][j].set_ylim(0, 1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sats_chisq_hist.png')
    plt.close()

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(12, 14), sharey='row', sharex='col')

    for i in range(len(lines)):

        results_file = f'/disk04/sapple/data/satellites/results/{model}_{wind}_{snap}_log_frad_1_fit_lines_{lines[i]}.h5'

        for j in range(nbins_fr200):

            with h5py.File(results_file, 'r') as hf:
                all_chisq = hf[f'chisq_{fr200[j]}r200'][:]

            mask = np.abs(np.log10(all_chisq)) < 2.

            ax[i][j].hist(np.log10(all_chisq[mask]), bins=100, density=True, cumulative=True, alpha=0.6)
            ax[i][j].axvline(np.log10(2.5), ls='--', c='k')
            if i == len(lines) -1:
                ax[i][j].set_xlabel(r'${\rm log}\ \chi^2_r$')
            if j == 0:
                ax[i][j].set_ylabel(r'${\rm Frequency}$')
                ax[i][j].annotate(plot_lines[i], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=13)
            if i == 0:
                ax[i][j].set_title(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[j]))
            ax[i][j].set_xlim(-2, 2)
            ax[i][j].set_ylim(0, 1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sats_chisq_cum_hist.png')
    plt.close()
