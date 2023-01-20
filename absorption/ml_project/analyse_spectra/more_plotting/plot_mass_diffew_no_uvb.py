import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def make_color_list(cmap, nbins):
    dc = 0.9 / (nbins -1)
    frac = np.arange(0.05, 0.95+dc, dc)
    return [cmap(i) for i in frac]

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$',
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['med', 'per25', 'per75']
    norients = 8
    fr200 = 0.25
    minT = ['4.0', '4.5', '5.0', '5.5', '6.0']

    delta_m = 0.25
    min_m = 10.
    nbins_m = 5

    colors = make_color_list(plt.get_cmap('viridis'), nbins_m)

    plot_dir = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    results_dir = f'/disk04/sapple/data/collisional/results/'
    sample_dir = f'/disk04/sapple/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        mass_long = np.repeat(sf['mass'][:], norients)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='row')
    ax = ax.flatten()

    for l, line in enumerate(lines):

        median_file = f'{results_dir}{model}_{wind}_{snap}_{line}_{fr200}r200_uvb_median_mass_ew.h5'
        if os.path.isfile(median_file):
            plot_data = read_h5_into_dict(median_file)

        else:

            plot_data = {}

            mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
            plot_data['mass'] = get_bin_middle(mass_bins)
            for i in range(len(minT)):
                for pq in plot_quantities:
                    plot_data[f'minT_{minT[i]}_{pq}'] = np.zeros(nbins_m)


            for i in range(len(minT)):

                no_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_no_uvb_minT_{minT[i]}_ew_{line}.h5')
                with_uvb_dict = read_h5_into_dict(f'{results_dir}{model}_{wind}_{snap}_with_uvb_minT_{minT[i]}_ew_{line}.h5')

                ew_no_uvb = no_uvb_dict[f'ew_wave_{fr200}r200'].flatten()
                ew_with_uvb = with_uvb_dict[f'ew_wave_{fr200}r200'].flatten()

                for j in range(nbins_m):

                    mask = (mass_long > mass_bins[j]) & (mass_long < mass_bins[j+1])
                    plot_data[f'minT_{minT[i]}_med'][j] = np.nanmedian(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]))
                    plot_data[f'minT_{minT[i]}_per25'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 25.)
                    plot_data[f'minT_{minT[i]}_per75'][j] = np.nanpercentile(np.log10(ew_no_uvb[mask]) - np.log10(ew_with_uvb[mask]), 75.)

            write_dict_to_h5(plot_data, median_file)



        for i in range(len(minT)):
            ax[l].plot(plot_data['mass'], plot_data[f'minT_{minT[i]}_med'], ls='-', c=colors[i], label=r'$T_{{\rm min}} = {{{}}}$'.format(minT[i]))
            if minT[i] == '5.0':
                ax[l].fill_between(plot_data['mass'], plot_data[f'minT_{minT[i]}_per25'], plot_data[f'minT_{minT[i]}_per75'], color=colors[i], alpha=0.4)
    
        if l == 0:
            ax[l].legend(fontsize=13, loc=3)
        if l in [3, 4, 5]:
            ax[l].set_xlabel(r'$\log\ (M_{*} / M_{\odot})$')
        if l in [0, 3]:
            ax[l].set_ylabel(r'${\rm log }( {\rm EW}_{\rm no UVB} / {\rm EW}_{\rm UVB} )$')
        ax[l].set_ylim(-1.5, 1.1)
        ax[l].annotate(plot_lines[l], xy=(0.05, 0.85), xycoords='axes fraction')
        ax[l].axhline(0., c='k', ls='--', lw=1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_uvb_test_{fr200}r200_mass_median_delta_ew.png')
    plt.show()
    plt.clf()
