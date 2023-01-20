# Didn't end up using this but considered saving a logfile of the fits that are already complete 

import os
import sys
import numpy as np
from spectrum import Spectrum

def read_last_line(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
        return int(last_line) +1
    else:
        return 0

def add_i_to_file(filename, i):
    with open(filename, 'a') as f:
        f.write(f'{i}\n')


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    logfile = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/logfile.txt'
    i = read_last_line(logfile)
    add_i_to_file(logfile, i)

    vel_range = 600.
    chisq_asym_thresh = -3.

    spec_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    try:
        spec_file = sorted(os.listdir(spec_dir))[1:][i]
    except IndexError:
        os.mknod(f'{spec_dir}done.txt')
        sys.exit()

    print(spec_file)

    spec = Spectrum(f'{spec_dir}{spec_file}')
    if hasattr(spec, 'line_list'):
        sys.exit()
    else:
        spec.main(vel_range=vel_range, chisq_asym_thresh=-3., write_lines=True)
