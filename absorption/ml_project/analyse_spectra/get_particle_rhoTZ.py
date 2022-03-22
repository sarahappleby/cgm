import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import pygad as pg
from pygadgetreader import readsnap

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}_phase_space.h5'

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    h = s.cosmology.h_0
    zsolar = 0.0134 # in Zo
    mh = 1.67372375e-27 # in kg

    lines = ["MgII2796", "CII1334", "SiIII1206", "OVI1031"]
    gizmo_index = [6, 2, 7, 4]
    elements = [pg.analysis.absorption_spectra.lines[line]['element'] for line in lines]
    ion_mass = np.array([pg.UnitArr(pg.analysis.absorption_spectra.lines[line]['atomwt']) * pg.physics.m_u for line in lines])

    n_bins = np.arange(-6, 0+0.1, 0.1)
    over_bins = np.arange(-1, 5, 0.1)
    temp_bins = np.arange(3, 8+0.1, 0.1) 

    gas_mass = np.log10(readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h) # in log Mo
    gas_metal_frac = readsnap(snapfile, 'Metallicity', 'gas', suppress=1, units=1) # in fraction 
    gas_temp = np.log10(readsnap(snapfile, 'u', 'gas', suppress=1, units=1)) # in log K
    gas_rho = np.log10(readsnap(snapfile, 'rho', 'gas', suppress=1, units=1)) # in log g/cm^3

    gas_overdensity = gas_rho - np.log10(cosmic_rho)
    # physical overdensity
    hist = plt.hist2d(gas_overdensity, gas_temp, bins=[over_bins, temp_bins], density=True, cmap='Greys')
    with h5py.File(results_file, 'a') as hf:
        hf.create_dataset('rho_overdensity_temp', data=np.array(np.rot90(hist[0])))
        hf.create_dataset('rho_overdensity_bins', data=np.array(over_bins))
        hf.create_dataset('temp_bins', data=np.array(temp_bins))

    # mean number density
    gas_nh = 10**gas_rho / mh
    hist = plt.hist2d(np.log10(gas_nh), gas_temp, bins=[n_bins, temp_bins], density=True, cmap='Greys')
    with h5py.File(results_file, 'a') as hf:
        hf.create_dataset('nh_temp', data=np.array(np.rot90(hist[0])))
        hf.create_dataset('n_bins', data=np.array(n_bins))

