import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import h5py
import pygad as pg
import sys

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

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    Nlabels = [r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm MgII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$', 
               r'${\rm log }(N\ {\rm SiIII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CIV} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm OVI} / {\rm cm}^{-2})$']
    x = [0.83, 0.78, 0.81, 0.79, 0.79, 0.8]
    x = [0.04]* 6
    #chisq_lim = [4.5, 63.1, 20.0, 70.8, 15.8, 4.5] limits with old fitting procedure
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]

    dT = 0.1
    T_min = 3.
    T_max = 6.
    delta_rho_min = -1.
    delta_rho_max = 5
    ddelta = 0.2
    N_min = [12, 11, 12, 11, 12, 12]
    N_max = 18.
    dN = 0.2
    T_bins = np.arange(T_min, T_max+dT, dT)
    delta_rho_bins = np.arange(delta_rho_min, delta_rho_max+ddelta, ddelta)
    N_bins = np.arange(np.min(N_min), N_max+dN, dN)

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]		
    rho_labels = ['Inner CGM', 'Outer CGM']
    rho_ls = ['--', ':']
    rho_lw = [1, 2]
    ssfr_labels = ['Star forming', 'Green valley', 'Quenched']
    ssfr_colors = [cb_blue, cb_green, cb_red]
    
    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    phase_space_file = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}_phase_space.h5'
    with h5py.File(phase_space_file, 'r') as hf:
        delta_rho_hist = hf['rho_delta_hist'][:]
        temp_hist = hf['temp_hist'][:]

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    #### Overdensity histograms

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    ssfr_lines = []
    for i in range(len(ssfr_colors)):
        ssfr_lines.append(Line2D([0,1],[0,1], color=ssfr_colors[i], ls='-', lw=1))
    leg = ax[0][0].legend(ssfr_lines, ssfr_labels, loc=1, fontsize=12)
    ax[0][0].add_artist(leg)

    rho_lines = []		
    for i in range(len(rho_ls)):		
        rho_lines.append(Line2D([0,1],[0,1], color='dimgrey', ls=rho_ls[i], lw=rho_lw[i]))		
    leg = ax[0][0].legend(rho_lines, rho_labels, loc=2, fontsize=12)		
    ax[0][0].add_artist(leg)		

    i = 0
    j = 0

    bin_edges = delta_rho_bins[:-1] + delta_rho_bins[1] - delta_rho_bins[0]
    bin_edges = np.insert(bin_edges, 0, bin_edges[0])
    delta_rho_hist = np.insert(delta_rho_hist, 0, 0)

    for line in lines:

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        #ax[i][j].step(bin_edges, delta_rho_hist, c=ssfr_colors[0], lw=1, ls=rho_ls[0])
        
        all_delta_rho = np.array([])
        all_ids = np.array([])

        sf_median = np.zeros(len(inner_outer))
        gv_median = np.zeros(len(inner_outer))
        q_median = np.zeros(len(inner_outer))

        for k in range(len(inner_outer)):

            rho = []
            N = []
            chisq = []
            ids = []

            for l in range(len(inner_outer[k])):

                with h5py.File(results_file, 'r') as hf:
                    rho.extend(hf[f'log_rho_{inner_outer[k][l]}r200'][:])
                    N.extend(hf[f'log_N_{inner_outer[k][l]}r200'][:])
                    chisq.extend(hf[f'chisq_{inner_outer[k][l]}r200'][:])
                    ids.extend(hf[f'ids_{inner_outer[k][l]}r200'][:])

            rho = np.array(rho)
            N = np.array(N)
            chisq = np.array(chisq)
            ids = np.array(ids)

            mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
            delta_rho = rho[mask] - np.log10(cosmic_rho)
            ids = ids[mask]

            mask = (delta_rho > delta_rho_bins[0]) & (delta_rho < delta_rho_bins[-1])

            #ax[i][j].axvline(np.nanmedian(delta_rho[mask]), color=ssfr_colors[0], ls=rho_ls[k+1], lw=1)

            idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
            all_mass = mass[idx]
            all_ssfr = ssfr[idx]
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            sf_median[k] = np.nanmedian(delta_rho[sf_mask*mask])
            gv_median[k] = np.nanmedian(delta_rho[gv_mask*mask])
            q_median[k] = np.nanmedian(delta_rho[q_mask*mask])

            all_delta_rho = np.append(all_delta_rho, delta_rho)
            all_ids = np.append(all_ids, ids)

        idx = np.array([np.where(gal_ids == l)[0] for l in all_ids]).flatten()
        all_mass = mass[idx]
        all_ssfr = ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        ax[i][j].hist(all_delta_rho[sf_mask], bins=delta_rho_bins, stacked=True, density=True, color=ssfr_colors[0], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_delta_rho[gv_mask], bins=delta_rho_bins, stacked=True, density=True, color=ssfr_colors[1], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_delta_rho[q_mask], bins=delta_rho_bins, stacked=True, density=True, color=ssfr_colors[2], ls='-', lw=1, histtype='step')
       
        for k in range(len(inner_outer)):
            ax[i][j].axvline(sf_median[k], ymin=.8, color=ssfr_colors[0], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(gv_median[k], ymin=.8, color=ssfr_colors[1], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(q_median[k], ymin=.8, color=ssfr_colors[2], ls=rho_ls[k], lw=rho_lw[k])

        ax[i][j].set_xlim(delta_rho_min, delta_rho_max)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction', 
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
        
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }\delta$')
        
        ax_top = ax[i][j].secondary_xaxis('top')
        ax_top.set_xticks(range(0, 6), labels=[])

        if line in ["SiIII1206", 'CIV14548']:
            ax[i][j].set_xticks(range(-1, 5))
        elif line in ["OVI1031"]:
            ax[i][j].set_xticks(range(0, 6))

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel('Frequency')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    ax[0][0].set_yticks(np.arange(0.2, 1.4, 0.2))

    ax[0][0].set_ylim(0, 1.2)
    ax[1][0].set_ylim(0, 1.2)

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_delta_hist_chisqion.png')
    plt.close()

    #### Temperature histograms

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        #ax[i][j].step(T_bins[:-1] + T_bins[1] - T_bins[0], temp_hist, c=ssfr_colors[0], lw=1, ls='-')

        all_T = np.array([])
        all_ids = np.array([])

        sf_median = np.zeros(len(inner_outer))
        gv_median = np.zeros(len(inner_outer))
        q_median = np.zeros(len(inner_outer))

        for k in range(len(inner_outer)):

            T = []
            N = []
            chisq = []
            ids = []

            for l in range(len(inner_outer[k])):

                with h5py.File(results_file, 'r') as hf:
                    T.extend(hf[f'log_T_{inner_outer[k][l]}r200'][:])
                    N.extend(hf[f'log_N_{inner_outer[k][l]}r200'][:])
                    chisq.extend(hf[f'chisq_{inner_outer[k][l]}r200'][:])
                    ids.extend(hf[f'ids_{inner_outer[k][l]}r200'][:])

            T = np.array(T)
            N = np.array(N)
            chisq = np.array(chisq)
            ids = np.array(ids)

            mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
            T = T[mask]
            ids = ids[mask]

            mask = (T > T_bins[0]) & (T < T_bins[-1])

            #ax[i][j].axvline(np.nanmedian(T[mask]), color=ssfr_colors[0], ls=rho_ls[k+1], lw=1)

            idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
            all_mass = mass[idx]
            all_ssfr = ssfr[idx]
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            sf_median[k] = np.nanmedian(T[sf_mask*mask])
            gv_median[k] = np.nanmedian(T[gv_mask*mask])
            q_median[k] = np.nanmedian(T[q_mask*mask])

            all_T = np.append(all_T, T)
            all_ids = np.append(all_ids, ids)

        idx = np.array([np.where(gal_ids == l)[0] for l in all_ids]).flatten()
        all_mass = mass[idx]
        all_ssfr = ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        ax[i][j].hist(all_T[sf_mask], bins=T_bins, stacked=True, density=True, color=ssfr_colors[0], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_T[gv_mask], bins=T_bins, stacked=True, density=True, color=ssfr_colors[1], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_T[q_mask], bins=T_bins, stacked=True, density=True, color=ssfr_colors[2], ls='-', lw=1, histtype='step')

        for k in range(len(inner_outer)):
            ax[i][j].axvline(sf_median[k], ymin=.8, color=ssfr_colors[0], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(gv_median[k], ymin=.8, color=ssfr_colors[1], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(q_median[k], ymin=.8, color=ssfr_colors[2], ls=rho_ls[k], lw=rho_lw[k])

        ax[i][j].set_xlim(T_min, T_max)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        ax_top = ax[i][j].secondary_xaxis('top')
        ax_top.set_xticks(np.arange(3, 6, 0.5), labels=[])

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log } (T / {\rm K})$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel('Frequency')

        if line in ["SiIII1206", 'CIV14548']:
            ax[i][j].set_xticks(np.arange(3, 6., 0.5))
        elif line in ["OVI1031"]:
            ax[i][j].set_xticks(np.arange(3.5, 6.5, 0.5))

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    ax[0][0].set_ylim(0, 2.3)
    ax[1][0].set_ylim(0, 2.3)

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_temp_hist_chisqion.png')
    plt.close()


    ### Column density histograms

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        #ax[i][j].step(T_bins[:-1] + T_bins[1] - T_bins[0], temp_hist, c=ssfr_colors[0], lw=1, ls='-')

        all_N = np.array([])
        all_ids = np.array([])

        sf_median = np.zeros(len(inner_outer))
        gv_median = np.zeros(len(inner_outer))
        q_median = np.zeros(len(inner_outer))

        for k in range(len(inner_outer)):

            N = []
            chisq = []
            ids = []

            for l in range(len(inner_outer[k])):

                with h5py.File(results_file, 'r') as hf:
                    N.extend(hf[f'log_N_{inner_outer[k][l]}r200'][:])
                    chisq.extend(hf[f'chisq_{inner_outer[k][l]}r200'][:])
                    ids.extend(hf[f'ids_{inner_outer[k][l]}r200'][:])

            N = np.array(N)
            chisq = np.array(chisq)
            ids = np.array(ids)

            mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
            N = N[mask]
            ids = ids[mask]

            mask = (N > N_bins[0]) & (N < N_bins[-1])

            #ax[i][j].axvline(np.nanmedian(T[mask]), color=ssfr_colors[0], ls=rho_ls[k+1], lw=1)

            idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
            all_mass = mass[idx]
            all_ssfr = ssfr[idx]
            sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

            sf_median[k] = np.nanmedian(N[sf_mask*mask])
            gv_median[k] = np.nanmedian(N[gv_mask*mask])
            q_median[k] = np.nanmedian(N[q_mask*mask])

            all_N = np.append(all_N, N)
            all_ids = np.append(all_ids, ids)

        idx = np.array([np.where(gal_ids == l)[0] for l in all_ids]).flatten()
        all_mass = mass[idx]
        all_ssfr = ssfr[idx]
        sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

        ax[i][j].hist(all_N[sf_mask], bins=N_bins, stacked=True, density=True, color=ssfr_colors[0], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_N[gv_mask], bins=N_bins, stacked=True, density=True, color=ssfr_colors[1], ls='-', lw=1, histtype='step')
        ax[i][j].hist(all_N[q_mask], bins=N_bins, stacked=True, density=True, color=ssfr_colors[2], ls='-', lw=1, histtype='step')

        for k in range(len(inner_outer)):
            ax[i][j].axvline(sf_median[k], ymin=.8, color=ssfr_colors[0], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(gv_median[k], ymin=.8, color=ssfr_colors[1], ls=rho_ls[k], lw=rho_lw[k])
            ax[i][j].axvline(q_median[k], ymin=.8, color=ssfr_colors[2], ls=rho_ls[k], lw=rho_lw[k])

        ax[i][j].set_xlim(N_min[lines.index(line)], N_max)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        ax_top = ax[i][j].secondary_xaxis('top')
        ax_top.set_xticks(range(0, 6), labels=[])

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(Nlabels[lines.index(line)])

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel('Frequency')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_N_hist_chisqion.png')
    plt.close()

