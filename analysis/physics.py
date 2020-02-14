import numpy as np
from cosmic_variance import *

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

def do_exclude_outliers(data_dict, out, both=False):
    if both:
        per_lo = np.nanpercentile(data_dict['sim_dist'], out)
    else:
        per_lo = 0.
    per_hi = np.nanpercentile(data_dict['sim_dist'], 100. - out)
    mask = (data_dict['sim_dist'] > per_lo) & (data_dict['sim_dist'] < per_hi)

    keys = list(data_dict.keys())
    for k in keys:
        data_dict[k] = data_dict[k][mask]
   
    return data_dict

def do_bins(x, nbins):

    rho_bins = get_bin_edges(x, nbins)
    plot_bins = get_bin_middle(rho_bins)

    return rho_bins, plot_bins

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr

def vel_to_wave(vel, lambda_rest, c, z):
        return lambda_rest * (1.0 + z) * (vel / c + 1.)

def wave_to_vel(wave, lambda_rest, c, z):
        return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)

def wave_to_z(wave, lambda_rest):
    return (wave - lambda_rest) / lambda_rest

def equivalent_width(flux, pixel_size):
    return np.sum((np.ones(len(flux)) - flux) * pixel_size)

def compute_binned_ew(ew, rho, rho_bins):
    binned_ew = bin_data(rho, ew, rho_bins)
    ew = np.array([np.nanmedian(i) for i in binned_ew])
    ew_low = np.array([np.nanpercentile(i, 25) for i in binned_ew])
    ew_high = np.array([np.nanpercentile(i, 75) for i in binned_ew])

    #sig_low = np.abs(ew - ew_low)
    #sig_high = np.abs(ew_high - ew)

    #ew, ewerr = convert_to_log(ew, np.array([sig_low, sig_high]))

    return ew, ew_low, ew_high

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

def sim_binned_cfrac(data_dict, mask, rho_bins, thresh, boxsize):
    binned_ew = bin_data(data_dict['sim_dist'][mask], data_dict['ew'][mask], rho_bins)
    binned_pos = bin_data(data_dict['sim_dist'][mask], data_dict['pos'][mask], rho_bins)

    cfrac = np.zeros(len(binned_ew))
    cfrac_cv = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        if len(binned_ew[i]) > 0.:
            cfrac[i], cfrac_cv[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'cfrac', thresh)
        else:
            cfrac[i], cfrac_cv[i] = np.nan, np.nan

    return cfrac, cfrac_cv

def cos_binned_cfrac(cos_dict, mask, rho_bins, thresh):
    binned_ew = bin_data(cos_dict['cos_dist'][mask], cos_dict['EW'][mask], rho_bins)
    cfrac = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        cfrac[i] = compute_cfrac(binned_ew[i], thresh)

    return cfrac

def compute_cfrac(ew, det_thresh):
    if len(ew) == 0.:
        return np.nan
    else:
        return float((len(np.where(ew > det_thresh)[0]))) / float(len(ew))

def compute_path_length(vgal, vel_window, lambda_rest, z, ):
    c = 2.99792e5 # km/s
    l_low = vel_to_wave(vgal - vel_window, lambda_rest, c, z)
    l_high = vel_to_wave(vgal + vel_window, lambda_rest, c, z)

    z_low = wave_to_z(l_low, lambda_rest)
    z_high = wave_to_z(l_high, lambda_rest)

    return np.abs(z_high - z_low)

def sim_binned_path_abs(data_dict, ssfr_mask, rho_bins, thresh, boxsize):
    ew_mask = data_dict['ew'] > thresh
    mask = ew_mask * ssfr_mask
    binned_ew = bin_data(data_dict['sim_dist'][mask], data_dict['ew'][mask], rho_bins)
    binned_pl = bin_data(data_dict['sim_dist'][mask], data_dict['path_length'][mask], rho_bins)
    binned_pos = bin_data(data_dict['sim_dist'][mask], data_dict['pos'][mask], rho_bins)
    
    path_abs = np.zeros(len(binned_ew))
    path_abs_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        if len(binned_ew[i]) > 0.:
            path_abs[i], path_abs_err[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'path_abs', pl=binned_pl[i])
        else:
            path_abs[i], path_abs_err[i] = np.nan, np.nan

    return path_abs, path_abs_err 

def cos_binned_path_abs(cos_dict, ssfr_mask, rho_bins, thresh):
    ew_mask = cos_dict['EW'] > thresh
    mask = ew_mask * ssfr_mask
    binned_ew = bin_data(cos_dict['cos_dist'][mask], cos_dict['EW'][mask], rho_bins)
    binned_pl = bin_data(cos_dict['cos_dist'][mask], cos_dict['path_length'][mask], rho_bins)

    path_abs = np.zeros(len(binned_ew))
    path_abs_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        path_abs[i] = compute_path_abs(binned_ew[i], binned_pl[i])

    return path_abs

def compute_path_abs(ew, pl):

    total_ew = np.nansum(ew)
    total_pl = np.nansum(pl)

    return np.divide(total_ew, total_pl, out=np.zeros_like(total_ew), where=total_pl!=0)


