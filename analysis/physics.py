import numpy as np
from cosmic_variance import *

def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])

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

def median_ew_cos_groups(ew, pos_arr, num_gals, num_cos, boxsize):
    new_ew = np.zeros(num_cos)
    err_cv = np.zeros(num_cos)
    err_std = np.zeros(num_cos)

    for i in range(num_cos):
        data = ew[i*num_gals:(i+1)*num_gals]
        data = np.sort(data)[1:num_gals - 1]
        pos = pos_arr[i*num_gals:(i+1)*num_gals]
        pos = np.sort(pos)[1:num_gals - 1]

        _, err_cv[i] = get_cosmic_variance(data, pos, boxsize, 'ew')    
        new_ew[i] = np.nanmedian(data)
        err_std[i] = np.nanstd(data)

    err = np.sqrt(err_std**2 + err_cv**2)
    new_ew, err = convert_to_log(new_err, err)

    return new_ew, err

def binned_cfrac(ew, rho, pos, rho_bins, thresh, boxsize):
    binned_ew = bin_data(rho, ew, rho_bins)
    binned_pos = bin_data(rho, pos, rho_bins)

    cfrac = np.zeros(len(binned_ew))
    cfrac_cv = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        cfrac[i], cfrac_cv[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'cfrac', thresh)

    return cfrac, cfrac_cv

def compute_cfrac(ew, det_thresh):
    if len(ew) == 0.:
        return np.nan
    else:
        return float((len(np.where(ew[i] > det_thresh)[0]))) / float(len(ew[i]))

def compute_path_length(vgal, vel_window, lambda_rest, z, ):
    c = 2.99792e5 # km/s
    l_low = vel_to_wave(vgal - vel_window, lambda_rest, c, z)
    l_high = vel_to_wave(vgal + vel_window, lambda_rest, c, z)

    z_low = wave_to_z(l_low, lambda_rest)
    z_high = wave_to_z(l_high, lambda_rest)

    return np.abs(z_high - z_low)

def binned_path_abs(ew, rho, pos, rho_bins, thresh, path_lengths, boxsize):
    mask = ew > thresh
    binned_ew = bin_data(rho[mask], ew[mask], rho_bins)
    binned_pl = bin_data(rho[mask], path_lengths[mask], rho_bins)
    binned_pos = bin_data(rho[mask], pos[mask], rho_bins)
    
    path_abs = np.zeros(len(binned_ew))
    path_abs_err = np.zeros(len(binned_ew))

    for i in range(len(binned_ew)):
        path_abs[i], path_abs_err[i] = get_cosmic_variance(binned_ew[i], binned_pos[i], boxsize, 'path_abs', binned_pl[i])

    return path_abs, path_abs_err 

def compute_path_abs(ew, pl):

    total_ew = np.nansum(ew)
    total_pl = np.nansum(pl)

    return np.divide(total_ew, total_pl, out=np.zeros_like(total_ew), where=total_pl!=0)


