import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import numpy as np
import h5py
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snaps = ['105', '125', '137', '151']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6], 
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}

    x = [0.04]* 6
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    ncells=16
    logN_min = 11.
    logN_max = 18.
    delta_logN = 0.5
    bins_logN = np.arange(logN_min, logN_max+delta_logN, delta_logN)
    bins_logN = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15., 16., 17., 18.])
    plot_logN = get_bin_middle(bins_logN)
    delta_N = np.array([10**bins_logN[i+1] - 10**bins_logN[i] for i in range(len(plot_logN))])

    idelta = 1. / (len(snaps) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('magma')
    cmap = truncate_colormap(cmap, 0.25, .9)
    redshift_colors = [cmap(i) for i in icolor]
    redshift_labels = [r'$z = 1$', r'$z = 0.5$', r'$z = 0.25$', r'$z = 0$']

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    redshift_lines = []
    for i in range(len(redshift_colors)):
        redshift_lines.append(Line2D([0,1],[0,1], color=redshift_colors[i]))
    leg = ax[0][0].legend(redshift_lines, redshift_labels, loc=1, fontsize=14)
    ax[0][0].add_artist(leg)

    i = 0
    j = 0

    for l, line in enumerate(lines):

        for s, snap in enumerate(snaps):

            chisq_lim = chisq_lim_dict[f'snap_{snap}']

            results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

            all_N = []
            all_ew = []
            all_chisq = []
            all_ids = []
            
            for k in range(len(fr200)):

                with h5py.File(results_file, 'r') as hf:
                    all_N.extend(hf[f'log_N_{fr200[k]}r200'][:])
                    all_ew.extend(hf[f'ew_{fr200[k]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{fr200[k]}r200'][:])
                    all_ids.extend(hf[f'ids_{fr200[k]}r200'][:])

            all_N = np.array(all_N)
            all_ew = np.array(all_ew)
            all_chisq = np.array(all_chisq)
            all_ids = np.array(all_ids)

            mask = (all_N > logN_min) * (all_chisq < chisq_lim[l]) * (all_ew >= 0.)
            all_N = all_N[mask]

            ax[i][j].hist(all_N, bins=bins_logN, color=redshift_colors[s], ls='-', lw=1, histtype='step')
        
        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.86), xycoords='axes fraction',
                          fontsize=12, bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }(N / {\rm cm}^{-2})$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'$n$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_redshift_Nhist.png')
    plt.show()
    plt.close()

