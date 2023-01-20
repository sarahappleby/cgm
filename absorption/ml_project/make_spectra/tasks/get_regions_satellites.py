# For the set of satellite-only spectra, don't perform the full fitting but run the region finding and return the number of 
# absorption regions in the LOS.

import os
import sys
import numpy as np
from spectrum import Spectrum

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    log_frad = sys.argv[4]
    i = int(sys.argv[5])

    vel_range = 600.
    chisq_asym_thresh = -3.
    chisq_unacceptable = 25.

    spec_dir = f'/disk04/sapple/data/satellites/{model}_{wind}_{snap}/log_frad_{log_frad}/'
    spec_file = sorted(os.listdir(spec_dir))[i]

    print(spec_file)

    spec = Spectrum(f'{spec_dir}{spec_file}')
    
    if hasattr(spec, 'line_list'):
        sys.exit()
    else:
        spec.main(vel_range=vel_range, do_regions=True, do_fit=False, plot_fit=False, write_lines=True)
