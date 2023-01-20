import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.lines import Line2D

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap

def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    plot_quantities = ['med', 'per25', 'per75', 'per16', 'per64']
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    log_frad = '1.0'

    delta_m = 0.5
    min_m = 10.
    nbins_m = 3
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    mass_bin_labels = []
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} M_\star < $' + f'{mass_bins[i+1]}')

    idelta = 1. / (len(mass_bins) -1)
    icolor = np.arange(0., 1.+idelta, idelta)
    cmap = cm.get_cmap('plasma')
    cmap = truncate_colormap(cmap, 0.2, .8)
    mass_colors = [cmap(i) for i in icolor]

    results_dir = f'/disk04/sapple/data/satellites/results/'
    normal_dir = f'/disk04/sapple/data/normal/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharey='row', sharex='col')
    ax = ax.flatten()

    mass_lines = []
    for i in range(len(mass_colors)):
        mass_lines.append(Line2D([0,1],[0,1], color=mass_colors[i]))
    leg = ax[0].legend(mass_lines, mass_plot_titles, loc=1, fontsize=13)
    ax[0].add_artist(leg)


    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_sat_median_ew_profile_mass.h5'
        normal_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')
        nlines_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_nlines_{line}.h5')
        sat_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_{log_frad}log_frad_ew_{line}.h5')

        for i in range(len(mass_plot_titles)):

            mass_mask = (mass_long > mass_bins[i]) & (mass_long < mass_bins[i+1])
            bin_label = mass_bin_labels[i]

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for pq in plot_quantities:
                plot_data[f'{bin_label}_{pq}'] = np.zeros(len(fr200))

            for j in range(len(fr200)):
                        
                ew_norm = normal_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                ew_sat = sat_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                nlines = nlines_dict[f'nlines_{fr200[j]}r200'].flatten()
                detect_mask = (nlines > 0.)

                plot_data[f'{bin_label}_med'][j] = np.nanmedian(np.log10(ew_sat[detect_mask*mass_mask] / ew_norm[detect_mask*mass_mask]))
                plot_data[f'{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_sat[detect_mask*mass_mask] / ew_norm[detect_mask*mass_mask]), 25.)
                plot_data[f'{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_sat[detect_mask*mass_mask] / ew_norm[detect_mask*mass_mask]), 75.)
                    
            ax[l].plot(plot_data['fr200'], plot_data[f'{bin_label}_med'], ls='-', c=mass_colors[i], lw=1.5)
            
            if i == 0:
                ax[l].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_per25'], plot_data[f'{bin_label}_per75'] , alpha=0.2, color=mass_colors[i])

            write_dict_to_h5(plot_data, profile_file)

        ax[l].set_ylim(-2, 0)
        
        if l == 0:
            ax[l].annotate(plot_lines[l], xy=(0.04, 0.88), xycoords='axes fraction',
                           bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
        else:
            ax[l].annotate(plot_lines[l], xy=(0.04, 0.06), xycoords='axes fraction',
                           bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }( {\rm EW}_{\rm sat} / {\rm EW}_{\rm total} )$')
        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$r_\perp / r_{200}$')
   
    ax[3].set_yticks(np.arange(-2, 0, 0.5))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sat_test_ew_profile_mass.pdf', format='pdf')
    plt.close()
