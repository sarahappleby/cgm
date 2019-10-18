# plots:
#   for each galaxy, each direction - do overlaid spectrum showing decrements of different ions
#   for each ion, 
#       equivalent width as a function of impact parameter
#       M* against SFR with coloured by EW of different ions


import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys

cos_survey = sys.argv[1]
ew_file = sys.argv[2]
plot_dir = sys.argv[3]

model = 'm50n512'
wind = 's50j7k'

cos_sample_file = '/home/sapple/cgm/cos_samples/'+cos_survey+'/samples/'+model+'_'+wind+'_'+cos_survey+'_galaxy_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    mass = np.repeat(f['mass'][:], 4)
    ssfr = np.log10(np.repeat(f['ssfr'][:], 4) + 1.e-14)

ions = ['H1215', 'MgII2796', 'SiII1260', 'CIV1548', 'OVI1031', 'NeVIII770']

for ion in ions:

    with h5py.File(ew_file, 'r') as f:
        ew = f[ion+'_ew'][:]

    plt.scatter(mass, ssfr, s=3, c=ew)
    plt.xlabel('M*')
    plt.ylabel('sSFR')
    plt.colorbar(label='EW '+ion)
    plt.savefig(plot_dir+'mass_ssfr_'+ion+'.png')
    plt.clf()

    plt.scatter(mass, ew, s=3, c=ssfr)
    plt.xlabel('M*')
    plt.ylabel('EW '+ion)
    plt.colorbar(label='sSFR')
    plt.savefig(plot_dir+'mass_'+ion+'.png')
    plt.clf()
