# Do a test spectrum generation comparing trident and pygad.

import sys
import os
import numpy as np
import h5py
import re
import yt
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg
from testing_generate_spectra import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Set some spectrum parameters:
sqrt2 = np.sqrt(2.)
snr = 30.
vel_range = 600. # km/s
pixel_size = 2.5 # km/s
periodic_vel = True
LSF = 'STIS_E140M'
fit_contin = True
Nbins = 8000

model = 'm100n1024'
wind = 's50'
snap = '151'

# for Mg:
gals = [1903, 5340]
fr200 = [0.25, 1.0]
orients = ['225', '45']

trident_lines = ['Ly a', 'Mg II 2796', 'C II 1334', 'Si III 1206']
pygad_lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206']
lambda_rest = [float(pg.analysis.absorption_spectra.lines[line]['l'].split(' ')[0]) for line in pygad_lines] 

plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CIV}1334$', r'${\rm SiIII}1206$']

snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/old_sample_files/{model}_{wind}_{snap}.hdf5'
s = pg.Snapshot(snapfile)
ds = yt.load(snapfile)

sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/old_sample_files/'
save_dir = f'/disk04/sapple/cgm/absorption/ml_project/make_spectra/trident_test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
    
    gal_ids = sf['gal_ids'][:]
    # we can't have the line of sight as a pygad UnitArr because it can't convert between kpc/h and ckpc/h_0
    # so instead we convert to default units of s['pos']
    # hence kpc/h and the factor of (1+z) is necessary
    gal_pos = (sf['position'][:]  *  (1.+s.redshift))
    gal_vel_pos = sf['vgal_position'][:][:, 2]
    gal_r200 = sf['halo_r200'][:] * (1.+s.redshift) # already in kpc/h, factor of 1+z for comoving

for i in range(len(gals)):

    index = int(np.where(gal_ids == gals[i])[0])
    gal_name = f'sample_galaxy_{gals[i]}_'

    pos = gal_pos[index]
    vel_pos = gal_vel_pos[index]
    rho = gal_r200[index] * fr200[i]

    if orients[i] == '0':
        los = pos[:2].copy(); los[0] += rho
    elif orients[i] == '45':
        los = pos[:2].copy(); los[0] += (rho / sqrt2); los[1] += (rho / sqrt2)
    elif orients[i] == '90':
        los = pos[:2].copy(); los[1] += rho
    elif orients[i] == '135':
        los = pos[:2].copy(); los[0] -= (rho / sqrt2); los[1] += (rho / sqrt2)
    elif orients[i] == '180':
        los = pos[:2].copy(); los[0] -= rho
    elif orients[i] == '225':
        los = pos[:2].copy(); los[0] -= (rho / sqrt2); los[1] -= (rho / sqrt2)
    elif orients[i] == '270':
        los = pos[:2].copy(); los[1] -= rho
    elif orients[i] == '315':
        los = pos[:2].copy(); los[0] += (rho / sqrt2); los[1] -= (rho / sqrt2)

    fig, ax = plt.subplots(len(pygad_lines), 1, figsize=(10, 10), sharey='row', sharex='col')
    ax = ax.flatten()

    for j in range(len(pygad_lines)):
        spec_name = f'{gal_name}{pygad_lines[j]}_{orients[i]}_deg_{fr200[i]}r200'
        generate_pygad_spectrum(s, los, pygad_lines[j], lambda_rest[j], vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}_pygad', LSF=LSF, fit_contin=fit_contin)
        generate_trident_spectrum(ds, trident_lines[j], los, f'{save_dir}{spec_name}_trident', lambda_rest[j], snr, Nbins)

        with h5py.File(f'{save_dir}{spec_name}_pygad.h5', 'r') as hf:
            ax[j].plot(hf['velocities'][:], np.log10(hf['taus'][:] + 1.e-5), label='Pygad', c='tab:pink')
        with h5py.File(f'{save_dir}{spec_name}_trident.h5', 'r') as hf:
            ax[j].plot(-1.*hf['velocities'][:], np.log10(hf['taus'][:] + 1.e-5), label='Trident', c='tab:blue')

        ax[j].annotate(plot_lines[j], xy=(0.05, 0.85), xycoords='axes fraction')
        ax[j].set_xlim(vel_pos - vel_range, vel_pos + vel_range)
        ax[j].set_ylim(-5.5, 1)

        if j == 0:
            ax[j].legend()
        
        if j == len(pygad_lines) -1:
            ax[j].set_xlabel(r'$v\ ({\rm km/s})$')

        ax[j].set_ylabel(r'${\rm log }\ \tau$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{save_dir}{gal_name}{orients[i]}_deg_{fr200[i]}r200_tau.png')
    plt.clf()

    fig, ax = plt.subplots(len(pygad_lines), 1, figsize=(10, 10), sharey='row', sharex='col')
    ax = ax.flatten()

    for j in range(len(pygad_lines)):
        spec_name = f'{gal_name}{pygad_lines[j]}_{orients[i]}_deg_{fr200[i]}r200'

        with h5py.File(f'{save_dir}{spec_name}_pygad.h5', 'r') as hf:
            ax[j].plot(hf['velocities'][:], hf['fluxes'][:], label='Pygad', c='tab:pink')
        with h5py.File(f'{save_dir}{spec_name}_trident.h5', 'r') as hf:
            ax[j].plot(-1.*hf['velocities'][:], hf['fluxes'][:], label='Trident', c='tab:blue')

        ax[j].annotate(plot_lines[j], xy=(0.05, 0.05), xycoords='axes fraction')
        ax[j].set_xlim(vel_pos - vel_range, vel_pos + vel_range)
        ax[j].set_ylim(-0.1, 1.1)

        if j == 0:
            ax[j].legend(loc=4)

        if j == len(pygad_lines) -1:
            ax[j].set_xlabel(r'$v\ ({\rm km/s})$')

        ax[j].set_ylabel(r'$F$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{save_dir}{gal_name}{orients[i]}_deg_{fr200[i]}r200_flux.png')
    plt.clf()

