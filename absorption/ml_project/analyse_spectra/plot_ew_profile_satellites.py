import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


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

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$', 
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['med', 'per25', 'per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    log_frad = ['0.0', '0.5', '1.0', '1.5', '2.0'] 

    delta_m = 0.5
    min_m = 10.5
    nbins_m = 1
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    bin_label = '10.5-11.0'
    mass_title = f'{mass_bins[0]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[1]}'

    colors = make_color_list(plt.get_cmap('viridis'), len(log_frad))

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/satellites/results/'
    normal_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample_old.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(2, 3, figsize=(14, 13), sharey='row', sharex='col')
    ax = ax.flatten()

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_sat_median_ew_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
     
        else:

            normal_dict = read_h5_into_dict(f'{normal_dir}{model}_{wind}_{snap}_ew_{line}.h5')
            mask = (mass_long > mass_bins[0]) & (mass_long < mass_bins[1])
    
            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for i in range(len(log_frad)):
                for pq in plot_quantities:
                    plot_data[f'log_frad_{log_frad[i]}_{bin_label}_{pq}'] = np.zeros(len(fr200))

            for i in range(len(log_frad)):

                sat_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_{log_frad[i]}log_frad_ew_{line}.h5')

                for j in range(len(fr200)):
                        
                    ew_norm = normal_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                    ew_sat = sat_dict[f'ew_wave_{fr200[j]}r200'].flatten()
                        
                    plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'][j] = np.nanmedian(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]))
                    plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per25'][j] = np.nanpercentile(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]), 25.)
                    plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per75'][j] = np.nanpercentile(np.log10(ew_sat[mask]) - np.log10(ew_norm[mask]), 75)

            write_dict_to_h5(plot_data, profile_file)

        for i in range(len(log_frad)):

            ax[l].plot(plot_data['fr200'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_med'], 
                       ls='-', c=colors[i], label=r'${{\rm log}} f_{{r_{{\rm half}} \star}} = {{{}}}$'.format(log_frad[i]), lw=1.5)
            if i == 1:
                ax[l].fill_between(plot_data['fr200'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per75'], plot_data[f'log_frad_{log_frad[i]}_{bin_label}_per25'], 
                                      alpha=0.3, color=colors[i])

        ax[l].set_ylim(-1.5, 0)
        ax[l].annotate(plot_lines[l], xy=(0.05, 0.05), xycoords='axes fraction')

        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }( {\rm EW}_{\rm sat} / {\rm EW}_{\rm total} )$')
        if l == 0:
            ax[l].legend(loc=4)
        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$\rho / r_{200}$')
        if l ==1:
            ax[l].set_title('Profiles for galaxies with ' + mass_title)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_sat_test_ew_profile_mstar.png')
    plt.show()
    plt.clf()
