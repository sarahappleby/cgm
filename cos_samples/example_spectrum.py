import numpy as np
import h5py
import matplotlib.pyplot as plt
import pygad as pg

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

def vel_to_wave(vel):
    lambda_rest = 1215.
    c = 2.99792e5 # km/s
    z = 0.2
    return lambda_rest * (1.0 + z) * (vel / c + 1.)

def wave_to_vel(wave):
    lambda_rest = 1215.
    c = 2.99792e5 # km/s
    z = 0.2

    return c * ((wave / lambda_rest) / (1.0 + z) - 1.0)

sample_file = 'm100n1024/cos_halos/samples/m100n1024_s50_cos_halos_sample.h5'

with h5py.File(sample_file, 'r') as f:
    gal_id = f['gal_ids'][:][0]
    vgal = f['vgal_position'][:][0][2]

example_spectrum = 'm100n1024/cos_halos/spectra/sample_galaxy_'+str(gal_id)+'_x_minus.h5'

ion = 'H1215'
snr = 12.
sigma_noise = 1./snr

with h5py.File(example_spectrum) as f:
    waves = f[ion+'_wavelength'][:]
    vels = f['velocity'][:]
    flux = f[ion+'_flux'][:]

noise = np.random.normal(0.0, sigma_noise, len(waves))
noise_vector = np.asarray([sigma_noise] * len(noise))

f_conv, n_conv = pg.analysis.absorption_spectra.apply_LSF(waves, flux, noise_vector, grating='COS_G130M')
f_noise = f_conv + noise
contin = pg.analysis.absorption_spectra.fit_continuum(waves, f_noise, noise_vector, order=1, sigma_lim=1.5)
f_contin = f_noise / contin

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(vels, flux, lw=1)
secax = ax.secondary_xaxis('top', functions=(vel_to_wave, wave_to_vel))
secax.set_xlabel(r'$\lambda\ (\AA)$')
ax.axvline(vgal, c='k', ls='--')
ax.axvline(vgal - 300., c='k', ls='--')
ax.axvline(vgal + 300., c='k', ls='--')
ax.set_xlabel(r'$v\ (\textrm{km/s})$')
ax.set_ylabel(r'$F$')
ax.set_xlim(np.min(vels), np.max(vels))
ax.set_ylim(0, )
plt.savefig('plots/example_raw.png')
plt.clf()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(vels, f_conv, lw=1)
secax = ax.secondary_xaxis('top', functions=(vel_to_wave, wave_to_vel))
secax.set_xlabel(r'$\lambda\ (\AA)$')
ax.axvline(vgal, c='k', ls='--')
ax.axvline(vgal - 300., c='k', ls='--')
ax.axvline(vgal + 300., c='k', ls='--')
ax.set_xlabel(r'$v\ (\textrm{km/s})$')
ax.set_ylabel(r'$F$')
ax.set_xlim(np.min(vels), np.max(vels))
ax.set_ylim(0, )
plt.savefig('plots/example_conv.png')
plt.clf()

fig, ax = plt.subplots(constrained_layout=True)
secax = ax.secondary_xaxis('top', functions=(vel_to_wave, wave_to_vel))
secax.set_xlabel(r'$\lambda\ (\AA)$')
ax.plot(vels, f_noise, lw=1)
ax.axvline(vgal, c='k', ls='--')
ax.axvline(vgal - 300., c='k', ls='--')
ax.axvline(vgal + 300., c='k', ls='--')
ax.set_xlabel(r'$v\ (\textrm{km/s})$')
ax.set_ylabel(r'$F$')
ax.set_xlim(np.min(vels), np.max(vels))
ax.set_ylim(0, )
plt.savefig('plots/example_noise.png')
plt.clf()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(vels, f_contin, lw=1)
secax = ax.secondary_xaxis('top', functions=(vel_to_wave, wave_to_vel))
secax.set_xlabel(r'$\lambda\ (\AA)$')
ax.axvline(vgal, c='k', ls='--')
ax.axvline(vgal - 300., c='k', ls='--')
ax.axvline(vgal + 300., c='k', ls='--')
ax.set_xlabel(r'$v\ (\textrm{km/s})$')
ax.set_ylabel(r'$F$')
ax.set_xlim(np.min(vels), np.max(vels))
ax.set_ylim(0, )
plt.savefig('plots/example_contin.png')
plt.clf()


