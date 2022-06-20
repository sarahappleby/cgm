import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


def make_color_list(cmap, nbins):
    dc = 1 / (nbins -1)
    frac = np.arange(0, 1+dc, dc)
    return [cmap(i) for i in frac]


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}$', r'${\rm MgII}$', r'${\rm CII}$',
                  r'${\rm SiIII}$', r'${\rm CIV}$', r'${\rm OVI}$']
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]

    deltath = 2.046913
    Tth = 5.
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    rho_labels = ['Inner CGM', 'Outer CGM']
    phase_labels = ['Condensed', 'Diffuse', 'Hot Halo', 'WHIM']
    cmap = plt.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.2, .95)
    cmap = plt.get_cmap('plasma')
    cmap = truncate_colormap(cmap, 0.1, .9)
    colors = make_color_list(cmap, len(phase_labels))

    colors = ['#a50162', '#d362a4', '#ff9956', '#d52d00'] # sapphic colour palette

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    fig, ax = plt.subplots(2, 1, figsize=(7, 8.5), sharey='row', sharex='col')

    handles = []
    handles.append(plt.Rectangle((10,10), 0.8, 0.8, color='dimgrey', edgecolor='dimgrey', alpha=0.6))
    handles.append(plt.Rectangle((10,10), 0.8, 0.8, color='dimgrey', edgecolor='dimgrey', fill=False, hatch='///'))
    leg = ax[0].legend(handles, rho_labels, loc=1, fontsize=13)
    ax[0].add_artist(leg)

    handles = [plt.Rectangle((10,10), 0.8, 0.8, color=colors[i], alpha=0.55) for i in range(len(phase_labels))]
    leg = ax[1].legend(handles, phase_labels, loc=1, fontsize=13)
    ax[1].add_artist(leg)

    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_T = []
        all_rho = []
        all_N = []
        all_chisq = []
        all_r = []

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_T.extend(hf[f'log_T_{fr200[i]}r200'][:])
                all_rho.extend(hf[f'log_rho_{fr200[i]}r200'][:])
                all_N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                all_r.extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

        all_T = np.array(all_T)
        all_rho = np.array(all_rho)
        all_N = np.array(all_N)
        all_chisq = np.array(all_chisq)
        all_r = np.array(all_r)

        mask = (all_N > N_min[lines.index(line)]) * (all_chisq < chisq_lim[lines.index(line)])
        all_T = all_T[mask]
        all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
        all_N = all_N[mask]
        all_r = all_r[mask]

        if line == 'MgII2796':
            mask = all_T < 5.
            all_T = all_T[mask]
            all_delta_rho = all_delta_rho[mask]
            all_N = all_N[mask]
            all_r = all_r[mask]

        r200_mask = all_r < 0.75
        condensed = (all_T < Tth) & (all_delta_rho > deltath)
        hot_halo = (all_T > Tth) & (all_delta_rho > deltath)
        whim = (all_T > Tth) & (all_delta_rho < deltath)
        diffuse = (all_T < Tth) & (all_delta_rho < deltath)

        total_absorbers = len(all_N)

        print('Condensed; diffuse; hot halo; WHIM')

        inner_fracs = np.zeros(4)
        inner_fracs[0] = len(all_N[r200_mask*condensed])
        inner_fracs[1] = len(all_N[r200_mask*diffuse])
        inner_fracs[2] = len(all_N[r200_mask*hot_halo])
        inner_fracs[3] = len(all_N[r200_mask*whim])
        inner_fracs /= total_absorbers

        outer_fracs = np.zeros(4)
        outer_fracs[0] = len(all_N[~r200_mask*condensed])
        outer_fracs[1] = len(all_N[~r200_mask*diffuse])
        outer_fracs[2] = len(all_N[~r200_mask*hot_halo])
        outer_fracs[3] = len(all_N[~r200_mask*whim])
        outer_fracs /= total_absorbers
        
        print(line, inner_fracs, outer_fracs)
        ax[0].bar(np.arange(0.05, 0.95, 0.225) + l, inner_fracs, width=0.225, align='edge',
                  color=colors, edgecolor=colors, alpha=0.55)
        ax[0].bar(np.arange(0.05, 0.95, 0.225) + l, outer_fracs, width=0.225, align='edge', bottom=inner_fracs,
                  color=colors, edgecolor=colors, fill=False, hatch='///')

    print('\n')
    ax_right = ax[0].secondary_yaxis('right')
    ax_right.set_yticks(np.arange(0, 1.2, 0.2), labels=[])

    #ax.set_yscale('log')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel(r'$\sum n_{\rm phase} / \sum n_{\rm CGM}$')
    ax[0].set_xticks(np.arange(0.43, 6.43, 1), plot_lines)


    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

        all_T = []
        all_rho = []
        all_N = []
        all_chisq = []
        all_r = []

        for i in range(len(fr200)):

            with h5py.File(results_file, 'r') as hf:
                all_T.extend(hf[f'log_T_{fr200[i]}r200'][:])
                all_rho.extend(hf[f'log_rho_{fr200[i]}r200'][:])
                all_N.extend(hf[f'log_N_{fr200[i]}r200'][:])
                all_chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
                all_r.extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

        all_T = np.array(all_T)
        all_rho = np.array(all_rho)
        all_N = np.array(all_N)
        all_chisq = np.array(all_chisq)
        all_r = np.array(all_r)

        mask = (all_N > N_min[lines.index(line)]) * (all_chisq < chisq_lim[lines.index(line)])
        all_T = all_T[mask]
        all_delta_rho = all_rho[mask] - np.log10(cosmic_rho)
        all_N = all_N[mask]
        all_r = all_r[mask]
       
        if line == 'MgII2796':
            mask = all_T < 5.
            all_T = all_T[mask]
            all_delta_rho = all_delta_rho[mask]
            all_N = all_N[mask]
            all_r = all_r[mask]

        r200_mask = all_r < 0.75
        condensed = (all_T < Tth) & (all_delta_rho > deltath)
        hot_halo = (all_T > Tth) & (all_delta_rho > deltath)
        whim = (all_T > Tth) & (all_delta_rho < deltath)
        diffuse = (all_T < Tth) & (all_delta_rho < deltath)
        
        total_absorption = np.nansum(10**all_N)

        print('Condensed; diffuse; hot halo; WHIM')

        inner_fracs = np.zeros(4)
        inner_fracs[0] = np.nansum(10**all_N[r200_mask*condensed])
        inner_fracs[1] = np.nansum(10**all_N[r200_mask*diffuse])
        inner_fracs[2] = np.nansum(10**all_N[r200_mask*hot_halo])
        inner_fracs[3] = np.nansum(10**all_N[r200_mask*whim])
        inner_fracs /= total_absorption

        outer_fracs = np.zeros(4)
        outer_fracs[0] = np.nansum(10**all_N[~r200_mask*condensed])
        outer_fracs[1] = np.nansum(10**all_N[~r200_mask*diffuse])
        outer_fracs[2] = np.nansum(10**all_N[~r200_mask*hot_halo])
        outer_fracs[3] = np.nansum(10**all_N[~r200_mask*whim])
        outer_fracs /= total_absorption

        print(line, inner_fracs, outer_fracs)
        ax[1].bar(np.arange(0.05, 0.95, 0.225) + l, inner_fracs, width=0.225, align='edge',
                  color=colors, edgecolor=colors, alpha=0.55)
        ax[1].bar(np.arange(0.05, 0.95, 0.225) + l, outer_fracs, width=0.225, align='edge', bottom=inner_fracs,
                  color=colors, edgecolor=colors, fill=False, hatch='///')

    print('\n')
    #ax.set_yscale('log')
    #ax.set_ylim(7e-4, 6)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel(r'$\sum N_{\rm phase} / \sum N_{\rm CGM}$')
    ax[1].set_xticks(np.arange(0.43, 6.43, 1), plot_lines)
    ax[1].set_yticks(np.arange(0., 1., 0.2))

    ax_right = ax[1].secondary_yaxis('right')
    ax_right.set_yticks(np.arange(0, 1.2, 0.2), labels=[])

    fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.tight_layout()
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_phase_bar_nN_sapphic.pdf', format='pdf')
    plt.close()

