import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys

gal = sys.argv[1]

sample_dir = '/home/sapple/cgm/cos_samples/samples/'
save_dir = '/home/sapple/cgm/cos_samples/plots/'

los = ['x_plus', 'x_minus', 'y_plus', 'y_minus']

sample_file = sample_dir + 'm50n512_s50j7k_cos_galaxy_sample.h5'
with h5py.File(sample_file, 'r') as f:
    ids = f['gal_ids'].value
    where = np.where(ids == int(float(gal)))
    vgal = float(f['vgal_position'].value[where, 2])

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for i, l in enumerate(los):
    spectrum_file = sample_dir+'spectra/sample_galaxy_'+gal+'_'+l + '.h5'
    with h5py.File(spectrum_file, 'r') as f:
        vel = f['velocity'].value
        flux = f['H1215_flux'].value
    vel -= vgal

    axes[i].plot(vel, flux, '--', c='b', lw=1.)
    #plt.axvline(0., '--', c='k', lw=1.)
    axes[i].set_xlim(-600., 600.)
    axes[i].set_ylim(0., 1.)
    axes[i].set_xlabel('v (km/s)')
    axes[i].set_ylabel('Flux')

plt.savefig(save_dir+'gal_'+gal+'.png')
plt.clf()

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i, l in enumerate(los):
    spectrum_file = sample_dir+'spectra/sample_galaxy_'+gal+'_'+l + '.h5'
    with h5py.File(spectrum_file, 'r') as f:
        vel = f['velocity'].value
        flux = f['H1215_flux'].value
        noise = f['noise'].value
    vel -= vgal


    axes[i].plot(vel, flux + noise, '--', c='b', lw=1.)
    #plt.axvline(0., '--', c='k', lw=1.)
    axes[i].set_xlim(-600., 600.)
    axes[i].set_ylim(0., 1.)
    axes[i].set_xlabel('v (km/s)')
    axes[i].set_ylabel('Flux')

plt.savefig(save_dir+'gal_'+gal+'_noise.png')
plt.clf()

