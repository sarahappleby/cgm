# Our main pipeline script to run pygad on each LOS
# Run using sub_line_pipeline.sh and sub_pipeline.sh

import sys
import os
import numpy as np
import h5py
import re
import yt
from yt.units.yt_array import YTArray, YTQuantity
import pygad as pg
from testing_generate_spectra import *

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
gals = [389]
fr200 = [0.25]
orients = ['0']

trident_lines = ['Ly a', 'Mg II 2796', 'C II 1334', 'Si III 1206']
pygad_lines = ['H1215', 'MgII2796', 'CII1334', 'SiIII1206']
lambda_rest = [float(pg.analysis.absorption_spectra.lines[line]['l'].split(' ')[0]) for line in pygad_lines] 

snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
s = pg.Snapshot(snapfile)
ds = yt.load(snapfile)

sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
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

    for j in range(len(pygad_lines)):
        spec_name = f'{gal_name}{pygad_lines[j]}_{orients[i]}_deg_{fr200[i]}r200'
        generate_pygad_spectrum(s, los, pygad_lines[j], lambda_rest[j], vel_pos, periodic_vel, pixel_size, snr, f'{save_dir}{spec_name}_pygad', LSF=LSF, fit_contin=fit_contin)
        generate_trident_spectrum(ds, trident_lines[j], los, f'{save_dir}{spec_name}_trident', lambda_rest[j], snr, Nbins)
