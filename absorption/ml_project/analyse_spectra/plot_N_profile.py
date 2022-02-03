import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import caesar

sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import read_h5_into_dict, write_dict_to_h5

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


def ssfr_b_redshift(z):
    return 1.9*np.log10(1+z) - 7.7


def sfms_line(mstar, a=0.73, b=-7.7):
    # The definition of the SFMS from Belfiore+18 is:
    # log (SFR/Msun/yr) = 0.73 log (Mstar/Msun) - 7.33
    # With a scatter of sigma = 0.39 dex
    return mstar*a + b


def ssfr_type_check(z, mstar, sfr):

    ssfr_b = ssfr_b_redshift(z)
    sf_line = sfms_line(mstar, a=0.73, b=ssfr_b)
    q_line = sfms_line(mstar, a=0.73, b=ssfr_b - 1.)

    sf_mask = sfr > sf_line
    gv_mask = (sfr < sf_line) & (sfr > q_line)
    q_mask = sfr < q_line
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    snap = sys.argv[2]
    wind = sys.argv[3]

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    redshift = sim.simulation.redshift

    lines = ["H1215", "MgII2796", "SiIII1206", "CIV1548", "OVI1031", "NeVIII770"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm SiIII}1206$', 
                  r'${\rm CIV}1548$', r'${\rm OVI}1031$', r'${\rm NeVIII}770$']
    plot_quantities = ['sf_med', 'sf_per25', 'sf_per75', 'gv_med', 'gv_per25', 'gv_per75', 'q_med', 'q_per25', 'q_per75',]
    norients = 8
    delta_fr200 = 0.25 
    min_fr200 = 0.25 
    nbins_fr200 = 5 
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    chisq_lim = 2.5

    delta_m = 0.25
    min_m = 10.
    nbins_m = 5
    mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
    
    mass_bin_labels = [] 
    mass_plot_titles = []
    for i in range(nbins_m):
        mass_bin_labels.append(f'{mass_bins[i]}-{mass_bins[i+1]}')
        mass_plot_titles.append(f'{mass_bins[i]}'+ r'$ < \textrm{log} (M_* / M_{\odot}) < $' + f'{mass_bins[i+1]}')

    results_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        gal_sm = sf['mass'][:]
        gal_sfr = sf['sfr'][:]

    mass_long = np.repeat(gal_sm, norients)
    sfr_long = np.repeat(np.log10(gal_sfr + 1e-3), norients)
    sf_mask, gv_mask, q_mask = ssfr_type_check(redshift, mass_long, sfr_long)

    fig, ax = plt.subplots(len(lines), nbins_m, sharey='row', sharex='col')

    for l, line in enumerate(lines):

        profile_file = f'{results_dir}{line}_median_N_profile.h5'

        if os.path.isfile(profile_file):
            plot_data = read_h5_into_dict(profile_file)
        else:

            chisq_dict = read_h5_into_dict(f'{results_dir}fit_max_chisq_{line}.h5')    
            mask_dict = {}
            for key in chisq_dict.keys():
                mask_dict[f'{key}_mask'] = np.abs(chisq_dict[key]) < chisq_lim
            del chisq_dict 
            fitN_dict = read_h5_into_dict(f'{results_dir}fit_column_densities_{line}.h5')

            plot_data = {}
            plot_data['fr200'] = fr200.copy()
            for bl in mass_bin_labels:
                for pq in plot_quantities:
                    plot_data[f'{bl}_{pq}'] = np.zeros(len(fr200))

            for i, bin_label in enumerate(mass_bin_labels):

                mass_mask = (mass_long > mass_bins[i]) & (mass_long < mass_bins[i+1])

                for j in range(len(fr200)):
                    totalN = fitN_dict[f'log_totalN_{fr200[j]}r200'].flatten()
                    chisq_mask = mask_dict[f'max_chisq_{fr200[j]}r200_mask'].flatten()

                    plot_data[f'{bin_label}_sf_med'][j] = np.nanmedian(totalN[chisq_mask*sf_mask*mass_mask])
                    plot_data[f'{bin_label}_sf_per25'][j] = np.nanpercentile(totalN[chisq_mask*sf_mask*mass_mask], 25)
                    plot_data[f'{bin_label}_sf_per75'][j] = np.nanpercentile(totalN[chisq_mask*sf_mask*mass_mask], 75)

                    plot_data[f'{bin_label}_gv_med'][j] = np.nanmedian(totalN[chisq_mask*gv_mask*mass_mask])
                    plot_data[f'{bin_label}_gv_per25'][j] = np.nanpercentile(totalN[chisq_mask*gv_mask*mass_mask], 25)
                    plot_data[f'{bin_label}_gv_per75'][j] = np.nanpercentile(totalN[chisq_mask*gv_mask*mass_mask], 75)

                    plot_data[f'{bin_label}_q_med'][j] = np.nanmedian(totalN[chisq_mask*q_mask*mass_mask])
                    plot_data[f'{bin_label}_q_per25'][j] = np.nanpercentile(totalN[chisq_mask*q_mask*mass_mask], 25)
                    plot_data[f'{bin_label}_q_per75'][j] = np.nanpercentile(totalN[chisq_mask*q_mask*mass_mask], 75)

            write_dict_to_h5(plot_data, profile_file)


        for b, bin_label in enumerate(mass_bin_labels):

            ax[l][b].plot(plot_data['fr200'], plot_data[f'{bin_label}_sf_med'], ls='-', c='b')
            ax[l][b].fill_between(plot_data['fr200']plot_data[f'{bin_label}_sf_per75'], plot_data[f'{bin_label}_sf_per25'], alpha=0.5, c='b')

            if b == 0:
                ax[l][b].set_ylabel(r'${{\rm log}} N_{{{}}}$'.format(plot_lines[l]))
            if l == 0:
                ax[l][b].set_title(mass_plot_titles[b])
            if l == len(lines) -1:
                ax[l][b].set_xlabel(r'$\rho / r_{200}$')


    # plot: each row is a different ion
    each column is a different mass bin
    column density against distance, lines split into green valley, star forming and quenched

