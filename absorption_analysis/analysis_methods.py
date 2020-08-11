import numpy as np
import h5py
from cosmic_variance import *
from physics import *

def read_simulation_sample(model, wind, snap, survey, norients, lines, r200_scaled):

    data_dict = {}
    cos_sample_file = '/home/sapple/cgm/cos_samples/'+model+'/cos_'+survey+'/samples/'+model+'_'+wind+'_cos_'+survey+'_sample.h5'
    with h5py.File(cos_sample_file, 'r') as f:
        data_dict['mass'] = np.repeat(f['mass'][:], norients)
        data_dict['ssfr'] = np.repeat(f['ssfr'][:], norients)
        data_dict['pos'] = np.repeat(f['position'][:], norients, axis=0)
        data_dict['r200'] = np.repeat(f['halo_r200'][:], norients)
    data_dict['ssfr'][data_dict['ssfr'] < -11.5] = -11.5

    for line in lines:
        # Read in the equivalent widths of the simulation galaxies spectra
        ew_file = 'data/cos_'+survey+'_'+model+'_'+wind+'_'+snap+'_ew_data_lsf.h5'
        with h5py.File(ew_file, 'r') as f:
            data_dict['ew_'+line] = f[line+'_wave_ew'][:]

    return data_dict

def get_equal_bins(data_dict, survey, r200_scaled=False):

    if r200_scaled:
        if survey == 'halos':
            r_end = 1.
        elif survey == 'dwarfs':
            r_end = 1.4
        dr = .2
    else:
        r_end = 200.
        dr = 40.

    data_dict['dist_bins_q'] = np.arange(0., r_end+dr, dr)
    data_dict['dist_bins_sf'] = np.arange(0., r_end+dr, dr)
    data_dict['plot_bins_q'] = data_dict['dist_bins_q'][:-1] + 0.5*dr
    data_dict['plot_bins_sf'] = data_dict['dist_bins_sf'][:-1] + 0.5*dr

    return data_dict

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

def get_bin_edges(x, nbins):
    dx_bin = 100. / nbins
    edges = np.zeros(nbins + 1)
    for i in range(nbins):
        edges[i] = np.nanpercentile(x, i*dx_bin)
    edges[-1] = np.max(x)
    return edges

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def do_bins(x, nbins):

    rho_bins = get_bin_edges(x, nbins)
    plot_bins = get_bin_middle(rho_bins)

    return rho_bins, plot_bins

def do_exclude_outliers(data_dict, out, both=False):
    if both:
        per_lo = np.nanpercentile(data_dict['dist'], out)
    else:
        per_lo = 0.
    per_hi = np.nanpercentile(data_dict['dist'], 100. - out)
    mask = (data_dict['dist'] > per_lo) & (data_dict['dist'] < per_hi)

    keys = list(data_dict.keys())
    for k in keys:
        data_dict[k] = data_dict[k][mask]

    return data_dict

def median_ew_cos_groups(ew, rho, ssfr, num_gals, num_cos):
    new_ew = np.zeros(num_cos)
    per_25 = np.zeros(num_cos)
    per_75 = np.zeros(num_cos)
    med_r = np.zeros(num_cos)
    med_s = np.zeros(num_cos)

    for i in range(num_cos):
        data = ew[i*num_gals:(i+1)*num_gals]
        rho_data = rho[i*num_gals:(i+1)*num_gals]
        ssfr_data = ssfr[i*num_gals:(i+1)*num_gals]

        med_r[i] = np.nanmedian(rho_data)
        med_s[i] = np.nanmedian(ssfr_data)
        new_ew[i] = np.nanmedian(data)
        per_25[i] = np.nanpercentile(data, 25)
        per_75[i] = np.nanpercentile(data, 75)

    sig_lo = np.abs(new_ew - per_25)
    sig_hi = np.abs(per_75 - new_ew)

    new_ew, err = convert_to_log(new_ew, np.array([sig_lo, sig_hi]))

    return new_ew, err, med_r, med_s

def sim_binned_ew(data_dict, mask, rho_bins, line, boxsize):
    binned_ew = bin_data(data_dict['dist'][mask], data_dict['ew_'+line][mask], rho_bins)
    binned_pos = bin_data(data_dict['dist'][mask], data_dict['pos'][mask], rho_bins)

    ew = np.zeros(len(binned_ew))
    ew_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        if len(binned_ew[i]) > 0.:
            ew[i], ew_err[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'ew')
        else:
            ew[i], ew_err[i] = np.nan, np.nan

    return convert_to_log(ew, ew_err)

def cos_binned_ew(cos_dict, mask, rho_bins):
    binned_ew = bin_data(cos_dict['dist'][mask], cos_dict['EW'][mask], rho_bins)
    binned_ew_err = bin_data(cos_dict['dist'][mask], cos_dict['EWerr'][mask], rho_bins)

    ew = np.zeros(len(binned_ew))
    lo = np.zeros(len(binned_ew))
    hi = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        data = binned_ew[i]
        ew[i] = np.nanmedian(data)
        lo[i] = np.nanpercentile(data, 25)
        hi[i] = np.nanpercentile(data, 75)

    sig_lo = np.abs(ew - lo)
    sig_hi = np.abs(hi - ew)

    return convert_to_log(ew, np.array([sig_lo, sig_hi]))

def sim_binned_cfrac(data_dict, mask, rho_bins, thresh, boxsize):
    binned_ew = bin_data(data_dict['dist'][mask], data_dict['ew'][mask], rho_bins)
    binned_pos = bin_data(data_dict['dist'][mask], data_dict['pos'][mask], rho_bins)

    cfrac = np.zeros(len(binned_ew))
    cfrac_cv = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        if len(binned_ew[i]) > 0.:
            cfrac[i], cfrac_cv[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'cfrac', thresh)
        else:
            cfrac[i], cfrac_cv[i] = np.nan, np.nan

    return cfrac, cfrac_cv

def cos_binned_cfrac(cos_dict, mask, rho_bins, thresh):
    binned_ew = bin_data(cos_dict['dist'][mask], cos_dict['EW'][mask], rho_bins)
    cfrac = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        cfrac[i] = compute_cfrac(binned_ew[i], thresh)

    return cfrac

def sim_binned_path_abs(data_dict, mask, rho_bins, thresh, boxsize):
    binned_ew = bin_data(data_dict['dist'][mask], data_dict['ew'][mask], rho_bins)
    binned_pl = bin_data(data_dict['dist'][mask], data_dict['path_length'][mask], rho_bins)
    binned_pos = bin_data(data_dict['dist'][mask], data_dict['pos'][mask], rho_bins)

    path_abs = np.zeros(len(binned_ew))
    path_abs_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        mask_ew = binned_ew[i] > thresh
        data = binned_ew[i][mask_ew]
        if len(data) > 0.:
            path_abs[i], path_abs_err[i] = get_cosmic_variance(binned_ew[i][mask_ew], binned_pos[i][mask_ew], boxsize, 'path_abs', pl=binned_pl[i])
        else:
            path_abs[i], path_abs_err[i] = np.nan, np.nan

    path_abs[path_abs == 0.] = 10**1.6
    path_abs[np.isnan(path_abs)] = 10**1.6

    return convert_to_log(path_abs, path_abs_err)

def cos_binned_path_abs(cos_dict, ssfr_mask, rho_bins, thresh):
    ew_mask = cos_dict['EW'] > thresh
    mask = ew_mask * ssfr_mask
    binned_ew = bin_data(cos_dict['dist'][mask], cos_dict['EW'][mask], rho_bins)
    binned_pl = bin_data(cos_dict['dist'][mask], cos_dict['path_length'][mask], rho_bins)

    path_abs = np.zeros(len(binned_ew))
    path_abs_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        path_abs[i] = compute_path_abs(binned_ew[i], binned_pl[i])

    return path_abs

