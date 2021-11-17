import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pygad as pg
from generate_spectra import *

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    i = int(sys.argv[1])

    vel_range = 600.
    z = 0

    spec_dir = f'data/{model}_{wind}_{snap}/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    fit_spectrum(f'{spec_dir}{spec_file}', vel_range=vel_range, z=z)
