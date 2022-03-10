import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
import sys

import caesar

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    log_frad_min = 0.5
    log_frad_max = 3.0
    log_dfrad = 0.5
    log_frad = np.arange(log_frad_min, log_frad_max+log_dfrad, log_dfrad)

    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    boxsize = sim.simulation.boxsize.in_units('kpc/h')

    central = np.array([i.central for i in sim.galaxies])
    gal_pos = np.array([i.pos.in_units('unitary') for i in sim.galaxies])[~central]
    gal_rad = np.array([i.radii['stellar_half_mass'].in_units('kpc/h') for i in sim.galaxies])[~central] / boxsize

    x_pos = gal_pos[:, 0]
    y_pos = gal_pos[:, 1]

    fig, ax = plt.subplots(2, 3, sharey='row', sharex='col')
    ax = ax.flatten()

    for i in range(len(log_frad)):

        rad = gal_rad * 10**log_frad[i]

        patches = []
        for j in range(len(gal_rad)):
            patches.append( Circle( (x_pos[j], y_pos[j]), rad[j]) )

        p = PatchCollection(patches, alpha=0.25)
        ax[i].add_collection(p)

        ax[i].annotate(r'${{\rm log}} f_{{r_{{\rm half}} \star}} = {{{}}}$'.format(log_frad[i]), 
                       xy=(0.05, 0.05), xycoords='axes fraction', size=11, bbox=dict(boxstyle="round", fc="w"))

        if i in [3, 4, 5]:
            ax[i].set_xlabel('x (unitary)')
        if i in [0, 3]:
            ax[i].set_ylabel('y (unitary)')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig('satellite_only.png')
    plt.show()
