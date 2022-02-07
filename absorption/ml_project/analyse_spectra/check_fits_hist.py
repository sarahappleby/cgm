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
    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"] 
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$',
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/chisq_plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(12, 14), sharey='row')

    for i in range(len(lines)):
    
        chisq_file = f'{results_dir}{model}_{wind}_{snap}_fit_max_chisq_{lines[i]}.h5'
        chisq_dict = read_h5_into_dict(chisq_file)

        for j in range(nbins_fr200):

            ax[i][j].hist(np.log10(chisq_dict[f'chisq_{fr200[j]}r200']), bins=100, density=True, alpha=0.6)
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
    plt.savefig(f'{plot_dir}chisq_hist.png')
    plt.clf()

    fig, ax = plt.subplots(len(lines), len(fr200), figsize=(12, 14), sharey='row')

    for i in range(len(lines)):

        chisq_file = f'{results_dir}{model}_{wind}_{snap}_fit_max_chisq_{lines[i]}.h5'
        chisq_dict = read_h5_into_dict(chisq_file)

        for j in range(nbins_fr200):

            ax[i][j].hist(np.log10(chisq_dict[f'chisq_{fr200[j]}r200']), bins=100, density=True, cumulative=True, alpha=0.6)
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

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}chisq_cum_hist.png')
    plt.clf()
