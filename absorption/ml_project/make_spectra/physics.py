import pygad as pg
import numpy as np

def convert_to_log(y, yerr):
    yerr /= (y*np.log(10.))
    y = np.log10(y)
    return y, yerr


def get_total_column_density(logN, dlogN):
    totalN = np.sum(10**(logN))
    dtotalN = np.sqrt(np.sum(dlogN**2.) * (np.log(10.)**2.) * (totalN**2.))
    return convert_to_log(totalN, dtotalN)


def vel_to_wave(vel, lambda_rest, z, v_units='km/s'):
    c = pg.physics.c.in_units_of(v_units)
    return lambda_rest * (1.0 + z) * (vel / c + 1.)


def wave_to_vel(wave, lambda_rest, z, v_units='km/s'):
    c = pg.physics.c.in_units_of(v_units)
    return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)


def tau_to_flux(tau):
    return np.exp(-np.clip(tau, -30, 30))
