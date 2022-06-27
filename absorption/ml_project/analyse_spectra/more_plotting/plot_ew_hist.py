import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206', 'CIV1548', 'OVI1031']
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    x = [0.04]* 6

    norients = 8
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    dEW = 0.1
    EW_min = -3.
    EW_max = 0.8
    EW_bins = np.arange(EW_min, EW_max+dEW, dEW)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    spectra_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    i = 0
    j = 0

    for line in lines:

        sum_ew_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_ew_{line}.h5'
        sum_ew_dict = read_h5_into_dict(sum_ew_file)

        all_sum_data = np.zeros(( len(fr200), len(sum_ew_dict[f'ew_wave_0.25r200']) * norients))
        for k in range(len(fr200)):
            all_sum_data[k] = sum_ew_dict[f'ew_wave_{fr200[k]}r200'].flatten()

        all_sum_data[all_sum_data < 0] = 0
        EW = np.log10(all_sum_data + 1.e-3).flatten()

        ax[i][j].hist(EW, bins=EW_bins, stacked=True, density=True, color='dimgrey', ls='-', lw=1, histtype='step')

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.8), xycoords='axes fraction',
                          bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
        
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log (EW}/\AA)$')

        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel('Frequency')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_ew_hist.png')
    plt.show()
    plt.close()

