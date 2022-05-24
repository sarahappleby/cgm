import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar
from cosmic_variance import get_cosmic_variance_ew

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

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

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift
    boxsize = float(sim.simulation.boxsize.in_units('kpc/h'))

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    plot_quantities = ['med', 'per25', 'per75', 'per16', 'per64']
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    log_frad = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5'] 

    delta_m = 0.5
    min_m = 10.5
    nbins_m = 1
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    bin_label = '10.5-11.0'
    mass_title = f'{mass_bins[0]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[1]}'

    colors = make_color_list(truncate_colormap(plt.get_cmap('magma'), 0., 1.0), len(log_frad))

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/satellites/results/'
    normal_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharey='row', sharex='col')
    ax = ax.flatten()

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_sat_median_ew_profile.h5'

        normal_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')
        nlines_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_nlines_{line}.h5')
        mask = (mass_long > mass_bins[0]) & (mass_long < mass_bins[1])
    
        plot_data = {}
        plot_data['fr200'] = fr200.copy()
        for i in range(len(log_frad)):
            for pq in plot_quantities:
                plot_data[f'log_frad_{log_frad[i]}_{bin_label}_{pq}'] = np.zeros(len(fr200))
            if log_frad[i] == '1.0':
                plot_data[f'log_frad_{log_frad[i]}_{bin_label}_cv'] = np.zeros(len(fr200))

        for i in range(len(log_frad)):

            sat_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_{log_frad[i]}log_frad_ew_{line}.h5')

            for j in range(len(fr200)):
                        
                ew_norm = normal_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                ew_sat = sat_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                nlines = nlines_dict[f'nlines_{fr200[j]}r200'].flatten()
                detect_mask = (nlines > 0.)

                plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'][j] = np.nanmedian(np.log10(ew_sat[mask*detect_mask] / ew_norm[mask*detect_mask]))
                plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_sat[mask*detect_mask] / ew_norm[mask*detect_mask]), 25.)
                plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_sat[mask*detect_mask] / ew_norm[mask*detect_mask]), 75.)
                    
                if log_frad[i] == '1.0':
                    los = sat_dict[f'LOS_pos_{fr200[j]}r200']
                    # do cosmic variance errors
                    _, cv_sat = get_cosmic_variance_ew(ew_sat[mask], los[mask], boxsize, ncells=16)
                    _, cv_sat = convert_to_log(np.nanmedian(ew_sat[mask]), cv_sat) 
                    _, cv_norm = get_cosmic_variance_ew(ew_norm[mask], los[mask], boxsize, ncells=16)
                    _, cv_norm = convert_to_log(np.nanmedian(ew_norm[mask]), cv_norm)
                    plot_data[f'log_frad_{log_frad[i]}_{bin_label}_cv'][j] = np.sqrt(cv_sat**2. + cv_norm**2.)

        write_dict_to_h5(plot_data, profile_file)

        for i in range(len(log_frad)):

            if log_frad[i] == '1.0':
                ax[l].plot(plot_data['fr200'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'], 
                           ls='-', c=colors[i], label=r'${{\rm log}} f_{{r_{{\rm half}} \star}} = {{{}}}$'.format(log_frad[i]), lw=1.5)
                
                #low = plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'] - plot_data[f'log_frad_{log_frad[i]}_{bin_label}_cv']
                #high = plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'] + plot_data[f'log_frad_{log_frad[i]}_{bin_label}_cv']
                #ax[l].fill_between(plot_data['fr200'], low, high, alpha=0.2, color=colors[i])
                ax[l].fill_between(plot_data['fr200'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per25'] , plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per75'] , alpha=0.2, color=colors[i])
            else:
                ax[l].plot(plot_data['fr200'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'],
                           ls='--', c=colors[i], label=r'${{\rm log}} f_{{r_{{\rm half}} \star}} = {{{}}}$'.format(log_frad[i]), lw=1.5)


        ax[l].set_ylim(-2, 0)
        ax[l].annotate(plot_lines[l], xy=(0.04, 0.06), xycoords='axes fraction',
                       bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }( {\rm EW}_{\rm sat} / {\rm EW}_{\rm total} )$')
        if l == 1:
            ax[l].legend(loc=4, fontsize=14.5)
        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$\rho / r_{200}$')
        #if l ==1:
        #    ax[l].set_title('Profiles for galaxies with ' + mass_title)
   
    ax[3].set_yticks(np.arange(-2, 0, 0.5))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sat_test_ew_profile_mstar.pdf', format='pdf')
    plt.close()
