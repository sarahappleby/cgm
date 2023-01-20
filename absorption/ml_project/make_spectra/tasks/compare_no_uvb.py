# Plot a test spectrum with and without the UVB.

import matplotlib.pyplot as plt
from spectrum import Spectrum

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

line = 'H1215'
#line = 'OVI1031'

spectrum_file_uvb = f'/disk04/sapple/data/normal/m100n1024_s50_151/sample_galaxy_195_{line}_0_deg_0.25r200.h5'
spectrum_file_no_uvb = f'/disk04/sapple/data/no_uvb/m100n1024_s50_151/sample_galaxy_195_{line}_0_deg_0.25r200.h5'

spec_uvb = Spectrum(spectrum_file_uvb)
spec_no_uvb = Spectrum(spectrum_file_no_uvb)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax = ax.flatten()

ax[0].plot(spec_uvb.velocities, np.log10(spec_uvb.taus), label='Collisional + UVB')
ax[0].legend(loc=1)
ax[0].set_ylabel(r'$\tau $')
ax[1].plot(spec_no_uvb.velocities, np.log10(spec_no_uvb.taus), label='Collisional only')
ax[1].legend(loc=4)
ax[1].set_xlabel('Velocity (km/s)')
ax[1].set_ylabel(r'$\tau $')
if line == 'H1215':
    ax[0].set_ylim(-6, 6)
    ax[1].set_ylim(-6, 6)
elif line == 'OVI1031':
    ax[0].set_ylim(-15, 2)
    ax[1].set_ylim(-15, 2)
plt.savefig(f'spec_compare_tau_{line}.png')
plt.clf()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax = ax.flatten()

ax[0].plot(spec_uvb.velocities, spec_uvb.fluxes, label='Collisional + UVB')
ax[0].legend(loc=4)
ax[0].set_ylabel('Flux')
ax[1].plot(spec_no_uvb.velocities, spec_no_uvb.fluxes, label='Collisional only')
ax[1].legend(loc=4)
ax[1].set_xlabel('Velocity (km/s)')
ax[1].set_ylabel('Flux')
plt.savefig(f'spec_compare_flux_{line}.png')
plt.clf()
