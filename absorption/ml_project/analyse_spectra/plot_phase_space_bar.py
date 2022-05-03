import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import pygad as pg
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

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
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6],
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']
    N_min = [12., 11., 12., 11., 12., 12.]

    deltath = 2.046913
    Tth = 5.
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    cmap = plt.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.2, .95)
    colors = make_color_list(cmap, len(lines))
    rho_labels = ['Inner CGM', 'Outer CGM']
    phase_labels = ['Hot halo', 'Condensed', 'Diffuse', 'WHIM']

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    #fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')
    fig, ax = plt.subplots()

    handles = [plt.Rectangle((10,10), 0.8, 0.8, color=colors[l]) for l in range(len(lines))]
    leg = ax.legend(handles, plot_lines, loc=2, fontsize=12)
    ax.add_artist(leg)

    handles = []
    handles.append(plt.Rectangle((10,10), 0.8, 0.8, color='dimgrey', edgecolor='dimgrey', alpha=0.6))
    handles.append(plt.Rectangle((10,10), 0.8, 0.8, color='dimgrey', edgecolor='dimgrey', fill=False, hatch='///'))
    leg = ax.legend(handles, rho_labels, loc=1, fontsize=12)
    ax.add_artist(leg)

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

        r200_mask = all_r < 1.0
        condensed = (all_T < Tth) & (all_delta_rho > deltath)
        hot_halo = (all_T > Tth) & (all_delta_rho > deltath)
        whim = (all_T > Tth) & (all_delta_rho < deltath)
        diffuse = (all_T < Tth) & (all_delta_rho < deltath)
        
        """
        total_absorption = np.nansum(10**all_N)

        inner_fracs = np.zeros(4)
        inner_fracs[0] = np.nansum(10**all_N[r200_mask*hot_halo])
        inner_fracs[1] = np.nansum(10**all_N[r200_mask*condensed])
        inner_fracs[2] = np.nansum(10**all_N[r200_mask*diffuse])
        inner_fracs[3] = np.nansum(10**all_N[r200_mask*whim])
        inner_fracs /= total_absorption

        outer_fracs = np.zeros(4)
        outer_fracs[0] = np.nansum(10**all_N[~r200_mask*hot_halo])
        outer_fracs[1] = np.nansum(10**all_N[~r200_mask*condensed])
        outer_fracs[2] = np.nansum(10**all_N[~r200_mask*diffuse])
        outer_fracs[3] = np.nansum(10**all_N[~r200_mask*whim])
        outer_fracs /= total_absorption

        """        
        total_absorption = np.nansum(all_N)

        inner_fracs = np.zeros(4)
        inner_fracs[0] = np.nansum(all_N[r200_mask*hot_halo])
        inner_fracs[1] = np.nansum(all_N[r200_mask*condensed])
        inner_fracs[2] = np.nansum(all_N[r200_mask*diffuse])
        inner_fracs[3] = np.nansum(all_N[r200_mask*whim])
        inner_fracs /= total_absorption

        outer_fracs = np.zeros(4)
        outer_fracs[0] = np.nansum(all_N[~r200_mask*hot_halo])
        outer_fracs[1] = np.nansum(all_N[~r200_mask*condensed])
        outer_fracs[2] = np.nansum(all_N[~r200_mask*diffuse])
        outer_fracs[3] = np.nansum(all_N[~r200_mask*whim])
        outer_fracs /= total_absorption

        ax.bar(np.arange(len(phase_labels))+(l*0.13), inner_fracs, width=0.13, align='edge', 
               color=colors[l], edgecolor=colors[l], alpha=0.6)
        ax.bar(np.arange(len(phase_labels))+(l*0.13), outer_fracs, width=0.13, align='edge', bottom=inner_fracs, 
               color=colors[l], edgecolor=colors[l], fill=False, hatch='///')

    #ax.set_yscale('log')
    #ax.set_ylim(7e-4, 6)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r'$\sum {\rm log}N_{\rm phase} / \sum {\rm log}N_{\rm CGM}$')
    #ax.set_ylabel(r'$\sum N_{\rm phase} / \sum N_{\rm CGM}$')
    ax.set_xticks(np.arange(0.43, 4.43, 1), phase_labels)
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_phase_bar_logN.png')
    #plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_phase_bar_N.png')
    plt.close()
