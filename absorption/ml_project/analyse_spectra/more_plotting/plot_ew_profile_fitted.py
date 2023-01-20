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

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift
    quench = quench_thresh(redshift)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    plot_quantities = ['sf_med', 'sf_per25', 'sf_per75', 'gv_med', 'gv_per25', 'gv_per75', 'q_med', 'q_per25', 'q_per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 2.5
    logN_min = 11.

    delta_m = 0.25
    min_m = 10.
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    
    mass_bin_labels = [] 
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    results_dir = f'/disk04/sapple/data/normal/results/'
    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]

    fig, ax = plt.subplots(len(lines), nbins_m, figsize=(14, 13), sharey='row', sharex='col')

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{model}_{wind}_{snap}_{line}_median_fitted_ew_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
        else:

            results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_ew_{line}.h5'

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for bl in mass_bin_labels:
                for pq in plot_quantities:
                    plot_data[f'{bl}_{pq}'] = np.zeros(len(fr200))

            for i in range(len(fr200)):

                with h5py.File(results_file, 'r') as hf:
                    all_N = hf[f'log_N_{fr200[i]}r200'][:]
                    all_ew = hf[f'ew_{fr200[i]}r200'][:]
                    all_chisq = hf[f'chisq_{fr200[i]}r200'][:]        
                    all_ids = hf[f'ids_{fr200[i]}r200'][:]
                        
                mask = (all_N > logN_min) * (all_chisq < chisq_lim)
                all_ew = all_ew[mask]
                all_ids = all_ids[mask]
                idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
                all_mass = mass[idx]
                all_ssfr = ssfr[idx]    

                sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

                for j, bin_label in enumerate(mass_bin_labels):
    
                    mass_mask = (all_mass > mass_bins[j]) & (all_mass < mass_bins[j+1])

                    plot_data[f'{bin_label}_sf_med'][i] = np.nanmedian(np.log10(all_ew[sf_mask*mass_mask]))
                    plot_data[f'{bin_label}_sf_per25'][i] = np.nanpercentile(np.log10(all_ew[sf_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_sf_per75'][i] = np.nanpercentile(np.log10(all_ew[sf_mask*mass_mask]), 75)

                    plot_data[f'{bin_label}_gv_med'][i] = np.nanmedian(np.log10(all_ew[gv_mask*mass_mask]))
                    plot_data[f'{bin_label}_gv_per25'][i] = np.nanpercentile(np.log10(all_ew[gv_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_gv_per75'][i] = np.nanpercentile(np.log10(all_ew[gv_mask*mass_mask]), 75)

                    plot_data[f'{bin_label}_q_med'][i] = np.nanmedian(np.log10(all_ew[q_mask*mass_mask]))
                    plot_data[f'{bin_label}_q_per25'][i] = np.nanpercentile(np.log10(all_ew[q_mask*mass_mask]), 25)
                    plot_data[f'{bin_label}_q_per75'][i] = np.nanpercentile(np.log10(all_ew[q_mask*mass_mask]), 75)

            write_dict_to_h5(plot_data, profile_file)


        for b, bin_label in enumerate(mass_bin_labels):

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_sf_med'], ls='-', c=cb_blue, label='SF', lw=1.5)
            if b == 0:
                ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_sf_per75'], plot_data[f'{bin_label}_sf_per25'], alpha=0.3, color=cb_blue)

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_gv_med'], ls='-', c=cb_green, label='GV', lw=1.5)
            if b == 0:
                ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_gv_per75'], plot_data[f'{bin_label}_gv_per25'], alpha=0.3, color=cb_green)

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_q_med'], ls='-', c=cb_red, label='Q', lw=1.5)
            if b == 0:
                ax[l][b].fill_between(plot_data['fr200'], plot_data[f'{bin_label}_q_per75'], plot_data[f'{bin_label}_q_per25'], alpha=0.3, color=cb_red)

            ax[l][b].set_ylim(-3., 0.)

            if b == 0:
                ax[l][b].set_ylabel(r'${\rm log (EW}/\AA)$')
            if b == nbins_m  -1:
                ax[l][b].annotate(plot_lines[l], xy=(0.05, 0.85), xycoords='axes fraction')
                if l == 0:
                    ax[l][b].legend(loc=1)
            if l == 0:
                ax[l][b].set_title(mass_plot_titles[b])
            if l == len(lines) -1:
                ax[l][b].set_xlabel(r'$\rho / r_{200}$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_fitted_ew_profile.png')
    plt.show()
    plt.clf()
