import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    i = int(sys.argv[4])

    vel_range = 600.
    chisq_asym_thresh = -3.
    chisq_unacceptable = 25.

    #spec_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}/'
    spec_dir = f'/disk04/sapple/data/normal/{model}_{wind}_{snap}_hm12/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    print(spec_file)

    spec = Spectrum(f'{spec_dir}{spec_file}')
    #spec = Spectrum(f'{spec_file}')
    
    if hasattr(spec, 'line_list'):
        sys.exit()
    else:
        spec.main(vel_range=vel_range, chisq_unacceptable=chisq_unacceptable, chisq_asym_thresh=chisq_asym_thresh, write_lines=True)
