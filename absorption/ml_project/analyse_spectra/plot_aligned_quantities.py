import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'


def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
    

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'

    line_a = 'H1215'
    line_b = ["CII1334", "CIV1548", "OVI1031"]
    plot_line_a = r'${\rm HI}1215$'
    plot_line_b = [r'${\rm CII}1334$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    #line_a = 'CII1334'
    #line_b = ["CIV1548", "OVI1031"]
    #plot_line_a = r'${\rm CII}1334$'
    #plot_line_b = [r'${\rm CIV}1548$', r'${\rm OVI}1031$']

    ion_mass_a = float(pg.UnitArr(pg.analysis.absorption_spectra.lines[line_a]['atomwt']) * pg.physics.m_u)
    ion_mass_b = np.array([pg.UnitArr(pg.analysis.absorption_spectra.lines[line]['atomwt']) * pg.physics.m_u for line in line_b])
    zsolar_a = 0.0134
    zsolar_b = [2.38e-3, 2.38e-3, 5.79e-3]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    quench = quench_thresh(redshift)
    chisq_lim = 2.5
    dv = 25

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    logN_min = 12.

    idelta = 0.8 / (len(fr200) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i) for i in icolor]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'


    # overdensity
    fig, ax = plt.subplots(4, len(line_b), figsize=(10, 10), sharey='row')

    fr200_points = []
    fr200_labels = []
    for i in range(len(fr200)):
        fr200_points.append(plt.scatter([0,1],[0,1],marker='o', color=colors[i]))
        fr200_labels.append(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
    leg = ax[0][0].legend(fr200_points, fr200_labels, loc=2, fontsize=12)
    ax[0][0].add_artist(leg)

    for l, line in enumerate(line_b):

        for i in range(len(fr200)):

            align_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b[l]}_{fr200[i]}r200.h5'

            with h5py.File(align_file, 'r') as hf:
                all_N_a = hf[f'{line_a}_log_N_{dv}kms'][:]
                all_rho_a = hf[f'{line_a}_log_rho_{dv}kms'][:]
                all_T_a = hf[f'{line_a}_log_T_{dv}kms'][:] 
                all_Z_a = hf[f'{line_a}_log_Z_{dv}kms'][:] - np.log10(zsolar_a)

                all_N_b = hf[f'{line_b[l]}_log_N_{dv}kms'][:]
                all_rho_b = hf[f'{line_b[l]}_log_rho_{dv}kms'][:]
                all_T_b = hf[f'{line_b[l]}_log_T_{dv}kms'][:]
                all_Z_b = hf[f'{line_b[l]}_log_Z_{dv}kms'][:] - np.log10(zsolar_b[l])

            mask = (all_N_a > logN_min) & (all_N_b > logN_min) 
            all_delta_rho_a = all_rho_a[mask] - np.log10(cosmic_rho)
            all_delta_rho_b = all_rho_b[mask] - np.log10(cosmic_rho)

            ax[0][l].scatter(all_N_a[mask], all_N_b[mask], color=colors[i], s=1.5)
            ax[1][l].scatter(all_delta_rho_a, all_delta_rho_b, color=colors[i], s=1.5)
            ax[2][l].scatter(all_T_a[mask], all_T_b[mask], color=colors[i], s=1.5)
            ax[3][l].scatter(all_Z_a[mask], all_Z_b[mask], color=colors[i], s=1.5)

        ax[0][l].set_xlim(logN_min, 17)
        ax[0][l].set_ylim(logN_min, 17)
        ax[1][l].set_xlim(-1, 5)
        ax[1][l].set_ylim(-1, 5)
        ax[2][l].set_xlim(3, 7)
        ax[2][l].set_ylim(3, 7)
        ax[3][l].set_xlim(-1, 0.5)
        ax[3][l].set_ylim(-1, 0.5)

        ax[0][l].set_title(plot_line_b[l]) 
        
        ax[0][l].set_xlabel(r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$')
        ax[1][l].set_xlabel(r'${\rm log }\Delta\ {\rm HI}$')
        ax[2][l].set_xlabel(r'${\rm log } (T\ {\rm HI} / {\rm K})$')
        ax[3][l].set_xlabel(r'${\rm log} (Z\ {\rm HI} / Z_{\odot})$')

        for i in range(4):
            ax[i][l].plot(np.arange(-20, 20),np.arange(-20, 20), ls=':', color='k', lw=1)

    ax[0][0].set_ylabel(r'${\rm log }(N\ {\rm Ion} / {\rm cm}^{-2})$')
    ax[1][0].set_ylabel(r'${\rm log }\Delta\ {\rm Ion}$')
    ax[2][0].set_ylabel(r'${\rm log } (T\ {\rm Ion} / {\rm K})$')
    ax[3][0].set_ylabel(r'${\rm log} (Z\ {\rm Ion} / Z_{\odot})$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.5)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_aligned_NdeltaTZ_{line_a}.png')
    plt.close()


    # number density
    fig, ax = plt.subplots(4, len(line_b), figsize=(10, 10), sharey='row')

    fr200_points = []
    fr200_labels = []
    for i in range(len(fr200)):
        fr200_points.append(plt.scatter([0,1],[0,1],marker='o', color=colors[i]))
        fr200_labels.append(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
    leg = ax[0][0].legend(fr200_points, fr200_labels, loc=2, fontsize=12)
    ax[0][0].add_artist(leg)

    for l, line in enumerate(line_b):

        for i in range(len(fr200)):

            align_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b[l]}_{fr200[i]}r200.h5'

            with h5py.File(align_file, 'r') as hf:
                all_N_a = hf[f'{line_a}_log_N_{dv}kms'][:]
                all_rho_a = hf[f'{line_a}_log_rho_{dv}kms'][:]
                all_T_a = hf[f'{line_a}_log_T_{dv}kms'][:]
                all_Z_a = hf[f'{line_a}_log_Z_{dv}kms'][:] - np.log10(zsolar_a)

                all_N_b = hf[f'{line_b[l]}_log_N_{dv}kms'][:]
                all_rho_b = hf[f'{line_b[l]}_log_rho_{dv}kms'][:]
                all_T_b = hf[f'{line_b[l]}_log_T_{dv}kms'][:]
                all_Z_b = hf[f'{line_b[l]}_log_Z_{dv}kms'][:] - np.log10(zsolar_b[l])

            mask = (all_N_a > logN_min) & (all_N_b > logN_min)
            all_n_a = np.log10(10**all_Z_a[mask] * 10**all_rho_a[mask]) - np.log10(ion_mass_a)
            all_n_b = np.log10(10**all_Z_b[mask] * 10**all_rho_b[mask]) - np.log10(ion_mass_b[l])

            ax[0][l].scatter(all_N_a[mask], all_N_b[mask], color=colors[i], s=1.5)
            ax[1][l].scatter(all_n_a, all_n_b, color=colors[i], s=1.5)
            ax[2][l].scatter(all_T_a[mask], all_T_b[mask], color=colors[i], s=1.5)
            ax[3][l].scatter(all_Z_a[mask], all_Z_b[mask], color=colors[i], s=1.5)

        ax[0][l].set_xlim(logN_min, 17)
        ax[0][l].set_ylim(logN_min, 17)
        ax[1][l].set_xlim(-6, 0)
        ax[1][l].set_ylim(-6, 0)
        ax[2][l].set_xlim(3, 7)
        ax[2][l].set_ylim(3, 7)
        ax[3][l].set_xlim(-1, 0.5)
        ax[3][l].set_ylim(-1, 0.5)

        ax[0][l].set_title(plot_line_b[l])

        ax[0][l].set_xlabel(r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$')
        ax[1][l].set_xlabel(r'${\rm log }(n\ {\rm HI} / {\rm cm}^{-3})$')
        ax[2][l].set_xlabel(r'${\rm log } (T\ {\rm HI} / {\rm K})$')
        ax[3][l].set_xlabel(r'${\rm log} (Z\ {\rm HI} / Z_{\odot})$')

        for i in range(4):
            ax[i][l].plot(np.arange(-20, 20),np.arange(-20, 20), ls=':', color='k', lw=1)

    ax[0][0].set_ylabel(r'${\rm log }(N\ {\rm Ion} / {\rm cm}^{-2})$')
    ax[1][0].set_ylabel(r'${\rm log }(n\ {\rm Ion} / {\rm cm}^{-3})$')
    ax[2][0].set_ylabel(r'${\rm log } (T\ {\rm Ion} / {\rm K})$')
    ax[3][0].set_ylabel(r'${\rm log} (Z\ {\rm Ion} / Z_{\odot})$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.5)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_aligned_NnTZ_{line_a}.png')
    plt.close()

