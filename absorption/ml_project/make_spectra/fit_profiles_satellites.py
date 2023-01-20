import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    i = int(sys.argv[4])

    log_frad = 1.0
    vel_range = 600.
    chisq_asym_thresh = -3.

    spec_dir = f'/disk04/sapple/data/satellites/{model}_{wind}_{snap}/log_frad_{log_frad}/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    print(spec_file)

    spec = Spectrum(f'{spec_dir}{spec_file}')
    if hasattr(spec, 'line_list'):
        sys.exit()
    else:
        spec.main(vel_range=vel_range, chisq_asym_thresh=-3., write_lines=True)
