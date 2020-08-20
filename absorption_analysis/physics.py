import numpy as np

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z - 9.

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

def compute_cfrac(ew, det_thresh):
    if len(ew) == 0.:
        cfrac = np.nan
        poisson = np.nan
    else:
        nlim = len(np.where(ew > det_thresh)[0])
        cfrac = float(nlim) / float(len(ew))
        poisson = np.sqrt(nlim) / float(len(ew))

    return cfrac, poisson

def compute_path_length(vgal, vel_window, lambda_rest, z, ):
    c = 2.99792e5 # km/s
    l_low = vel_to_wave(vgal - vel_window, lambda_rest, c, z)
    l_high = vel_to_wave(vgal + vel_window, lambda_rest, c, z)

    z_low = wave_to_z(l_low, lambda_rest)
    z_high = wave_to_z(l_high, lambda_rest)

    return np.abs(z_high - z_low)

def compute_path_abs(ew, pl):

    total_ew = np.nansum(ew)
    total_pl = np.nansum(pl)

    return np.divide(total_ew, total_pl, out=np.zeros_like(total_ew), where=total_pl!=0)

def propogate_path_abs_err(ew, ew_err, pl):
    total_pl = np.nansum(pl)
    total_sq_err = np.nansum(ew_err**2.)

    return np.sqrt(total_sq_err) / total_pl

