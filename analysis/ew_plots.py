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

model = 'm100n1024'
wind = 's50'

plot_dir = '/home/sapple/cgm/analysis/plots/cos_'+cos_survey+'/'
cos_sample_file = '/home/sapple/cgm/cos_samples/cos_'+cos_survey+'/samples/'+model+'_'+wind+'_cos_'+cos_survey+'_sample.h5'
with h5py.File(cos_sample_file, 'r') as f:
    mass = np.repeat(f['mass'][:], 4)
    ssfr = np.repeat(f['ssfr'][:], 4)



ions = ['H1215', 'MgII2796', 'SiIII1206', 'CIV1548', 'OVI1031', 'NeVIII770']

for ion in ions:

    with h5py.File(ew_file, 'r') as f:
        ew = np.log10(f[ion+'_wave_ew'][:])

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
