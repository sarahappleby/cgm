import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap

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

    snapfile = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    quench = quench_thresh(redshift)
    chisq_lim = 2.5
    dv = 10
    logN_min = 12.

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    cmap = cm.get_cmap('viridis')
    cmap = truncate_colormap(cmap, 0.1, 0.9)
    norm = colors.BoundaryNorm(np.arange(0.125, 1.625, 0.25), cmap.N)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'

    fig, ax = plt.subplots(3, len(line_b), figsize=(10, 10), sharey='row')


    #fr200_points = []
    #fr200_labels = []
    #for i in range(len(fr200)):
    #    fr200_points.append(plt.scatter([-10,-11],[-10,-11],marker='o', color=colors[i]))
    #    fr200_labels.append(r'$\rho / r_{{200}} = {{{}}}$'.format(fr200[i]))
    #leg = ax[0][0].legend(fr200_points, fr200_labels, loc=2, fontsize=12)
    #ax[0][0].add_artist(leg)

    for l, line in enumerate(line_b):

        all_dv = []
        all_N_a = []
        all_rho_a = []
        all_T_a = []
        all_N_b = []
        all_rho_b = []
        all_T_b = []
        all_r = []

        for i in range(len(fr200)):

            align_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_aligned_{line_a}_{line_b[l]}_chisq{chisq_lim}.h5'

            with h5py.File(align_file, 'r') as hf:
                all_dv.extend(hf[f'dv_{fr200[i]}'][:])

                all_N_a.extend(hf[f'{line_a}_log_N_{fr200[i]}'][:])
                all_rho_a.extend(hf[f'{line_a}_log_rho_{fr200[i]}'][:])
                all_T_a.extend(hf[f'{line_a}_log_T_{fr200[i]}'][:])

                all_N_b.extend(hf[f'{line_b[l]}_log_N_{fr200[i]}'][:])
                all_rho_b.extend(hf[f'{line_b[l]}_log_rho_{fr200[i]}'][:])
                all_T_b.extend(hf[f'{line_b[l]}_log_T_{fr200[i]}'][:])

                all_r.extend([fr200[i]] * len(hf[f'dv_{fr200[i]}'][:]))

        all_dv = np.array(all_dv)
        all_N_a = np.array(all_N_a)
        all_rho_a = np.array(all_rho_a)
        all_T_a = np.array(all_T_a)
        all_N_b = np.array(all_N_b)
        all_rho_b = np.array(all_rho_b)
        all_T_b = np.array(all_T_b)
        all_r = np.array(all_r)

        N_mask = (all_N_a > logN_min) & (all_N_b > logN_min) 
        dv_mask = all_dv < dv 
        mask = N_mask * dv_mask
        all_delta_rho_a = all_rho_a[mask] - np.log10(cosmic_rho)
        all_delta_rho_b = all_rho_b[mask] - np.log10(cosmic_rho)

        #ax[0][l].scatter(all_N_a[mask], all_N_b[mask], color=colors[i], s=1.5)
        #ax[1][l].scatter(all_delta_rho_a, all_delta_rho_b, color=colors[i], s=1.5)
        #ax[2][l].scatter(all_T_a[mask], all_T_b[mask], color=colors[i], s=1.5)

        plot_order = np.arange(len(all_N_a[mask]))
        np.random.shuffle(plot_order)

        im = ax[0][l].scatter(all_N_a[mask][plot_order], all_N_b[mask][plot_order], c=all_r[mask][plot_order], cmap=cmap, norm=norm, s=1.5)
        ax[1][l].scatter(all_delta_rho_a[plot_order], all_delta_rho_b[plot_order], c=all_r[mask][plot_order], cmap=cmap, norm=norm, s=1.5)
        ax[2][l].scatter(all_T_a[mask][plot_order], all_T_b[mask][plot_order], c=all_r[mask][plot_order], cmap=cmap, norm=norm, s=1.5)

        ax[0][l].set_xlim(logN_min, 17)
        ax[0][l].set_ylim(logN_min, 17)
        ax[1][l].set_xlim(-1, 5)
        ax[1][l].set_ylim(-1, 5)
        ax[2][l].set_xlim(3, 7)
        ax[2][l].set_ylim(3, 7)

        ax[0][l].set_title(plot_line_b[l]) 
        
        ax[0][l].set_xlabel(r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$')
        ax[1][l].set_xlabel(r'${\rm log }\Delta\ {\rm HI}$')
        ax[2][l].set_xlabel(r'${\rm log } (T\ {\rm HI} / {\rm K})$')

        #ax[0][l].set_xlabel(r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$')
        #ax[1][l].set_xlabel(r'${\rm log }\Delta\ {\rm CII}$')
        #ax[2][l].set_xlabel(r'${\rm log } (T\ {\rm CII} / {\rm K})$')

        for i in range(3):
            ax[i][l].plot(np.arange(-20, 20),np.arange(-20, 20), ls=':', color='k', lw=1)

    ax[0][0].set_ylabel(r'${\rm log }(N\ {\rm Ion} / {\rm cm}^{-2})$')
    ax[1][0].set_ylabel(r'${\rm log }\Delta\ {\rm Ion}$')
    ax[2][0].set_ylabel(r'${\rm log } (T\ {\rm Ion} / {\rm K})$')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=fr200, label=r'$\rho / r_{200}$')
    fig.subplots_adjust(wspace=0., hspace=0.3)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_aligned_NdeltaT_{line_a}.png')
    plt.close()

