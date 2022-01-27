import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'
    i = int(sys.argv[1])

    vel_range = 600.
    chisq_asym_thresh = -3.

    spec_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/{model}_{wind}_{snap}/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    spec = Spectrum(f'{spec_dir}{spec_file}')
    if hasattr(spec, 'line_list'):
        sys.exit()
    else:
        spec.main(vel_range=vel_range, chisq_asym_thresh=-3., write_lines=True)
