'''
Script to generate spectra through simulations using pygad and trident (yt).
Author: Kate Storey-Fisher
Date: June 20, 2017
'''

import numpy as np
import time
import pylab as plt
import sys

import pygad as pg
import yt
import trident
import caesar
import h5py
from astropy.cosmology import FlatLambdaCDM
from yt.utilities.cosmology import Cosmology

igal = 4
Nbins = 1000
snr = 1000.0
periodic_vel = True

c = pg.physics.c.in_units_of('km/s')

MODEL = sys.argv[1]
SNAP = int(sys.argv[2])
WIND = sys.argv[3]
igal = int(sys.argv[4])
snap_file = '/home/rad/data/%s/%s/snap_%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)
infile = '/home/rad/data/%s/%s/Groups/%s_%03d.hdf5' % (MODEL,WIND,MODEL,SNAP)

TINIT = time.time()
def t_elapsed(): return np.round(time.time()-TINIT,2)

def generate_pygad_spectrum(s, line_name, los, spec_name, lambda_rest):
    print("Generating pygad spectrum {}...".format(spec_name))

    H = s.cosmology.H(s.redshift).in_units_of('(km/s)/Mpc', subs=s)
    box_width = s.boxsize.in_units_of('Mpc', subs=s)
    vbox = H * box_width
    if periodic_vel: v_limits = [-0.5*vbox, 1.5*vbox]
    else: v_limits = [0, vbox]

    #taus, col_densities, temps, v_edges, phys_densities, restr_column = pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line_name, v_limits, Nbins=Nbins, periodic_box=True, unweighted=False)  # Kate's pygad version
    taus, col_densities, temps, v_edges, restr_column = pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line_name, v_limits, Nbins=Nbins)  # Standard version
    velocities = 0.5 * (v_edges[1:] + v_edges[:-1])
    wavelengths = lambda_rest * (s.redshift + 1) * (1 + velocities / c)
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    noise_vector = [sigma_noise] * len(noise)
    # periodically wrap velocities into the range [-vbox,0] -- should only be used when generating *random* LOS
    if periodic_vel: 
        npix_periodic = int( vbox / (max(velocities)-min(velocities)) * len(wavelengths) )
        print 'periodically wrapping optical depths with %d pixels'%npix_periodic
        for i in range(0,len(wavelengths)):
            if velocities[i] < -vbox: taus[i+npix_periodic] += taus[i]
            if velocities[i] > 0: taus[i-npix_periodic] += taus[i]
    #plt.plot(velocities,np.log10(taus),'-',c='r',label='Trident')
    fluxes = np.exp(-np.array(taus)) + noise
    print 'Pygad generated spectrum v=',v_edges[0],v_edges[-1]
    plt.xlim(0,vbox)
    plt.plot(velocities,fluxes,'--',c='r',label='Pygad')
    plt.plot(velocities,noise_vector,'--',c='y')

    # Save spectrum to hdf5 format
    with h5py.File('{}.h5'.format(spec_name), 'w') as hf:
        hf.create_dataset('velocity', data=np.array(velocities))
        hf.create_dataset('flux', data=np.array(fluxes))
        hf.create_dataset('wavelength', data=np.array(wavelengths))
        hf.create_dataset('tau', data=np.array(taus))
        hf.create_dataset('noise', data=np.array(noise_vector))
        hf.create_dataset('density_col', data=np.array(col_densities))
        #hf.create_dataset('density_phys', data=np.array(phys_densities))
        hf.create_dataset('temp', data=np.array(temps))
    print('Pygad spectrum done [t=%g s]'%(t_elapsed()))


def generate_trident_spectrum(ds, line_name, los, spec_name, lambda_rest):
    print("Generating trident spectrum...")
    # Generate ray through box
    ray_start = [los[0], los[1], ds.domain_left_edge[2]]
    ray_end = [los[0], los[1], ds.domain_right_edge[2]]
    line_list = [line_name]
    ray = trident.make_simple_ray(ds,
                                  start_position=ray_start,
                                  end_position=ray_end,
                                  data_filename="ray.h5",
                                  lines=line_list,
                                  ftype='PartType0')
    ar = ray.all_data()
    print 'ray=',ray,ar

    # Set up trident spectrum generator and generate specific spectrum using that ray
    co = Cosmology()
    print co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc"),ds.hubble_constant
    vbox = ds.domain_right_edge[2] * co.hubble_parameter(ds.current_redshift).in_units("km/s/kpc") / ds.hubble_constant / (1.+ds.current_redshift)
    if periodic_vel:
        lambda_min = lambda_rest * (1 + ds.current_redshift) * (1. - 1.5*vbox.value / c)
        lambda_max = lambda_rest * (1 + ds.current_redshift) * (1. + 0.5*vbox.value / c)
    else:
        lambda_min = lambda_rest * (1 + ds.current_redshift) * (1. - 1.0*vbox.value / c)
        lambda_max = lambda_rest * (1 + ds.current_redshift) * (1. + 0.0*vbox.value / c)
    print 'vbox=',vbox,lambda_rest,lambda_min,lambda_max
    sg = trident.SpectrumGenerator(lambda_min=lambda_min, lambda_max=lambda_max, n_lambda=Nbins)
    #sg = trident.SpectrumGenerator('COS-G130M')  # convolves with COS line spread fcn, gives COS resolution
    sg.make_spectrum(ray, lines=line_list)

    # Get fields and convert wavelengths to velocities. Note that the velocities are negative compared to pygad!
    wavelengths = np.array(sg.lambda_field)
    taus = np.array(sg.tau_field)
    sigma_noise = 1.0/snr
    noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    noise_vector = [sigma_noise] * len(noise)
    fluxes = np.array(sg.flux_field) + noise
    velocities = c * ((wavelengths / lambda_rest) / (1.0 + ds.current_redshift) - 1.0)
    print 'Trident generated spectrum v=',min(velocities),max(velocities),len(wavelengths),len(velocities)

    # periodically wrap velocities into the range [-vbox,0] -- should only be used when generating *random* LOS
    if periodic_vel: 
        npix_periodic = int( vbox / (max(velocities)-min(velocities)) * len(wavelengths) )
        print 'periodically wrapping optical depths with %d pixels'%npix_periodic
        for i in range(0,len(wavelengths)):
            if velocities[i] < -vbox: taus[i+npix_periodic] += taus[i]
            if velocities[i] > 0: taus[i-npix_periodic] += taus[i]
    #plt.plot(velocities,np.log10(taus),'-',c='r',label='Trident')
        fluxes = np.exp(-taus) + noise

    # plot spectrum
    plt.plot(-velocities,fluxes,':',c='g',label='Trident')
    plt.plot(-velocities,noise_vector,'--',c='c')

    #Save spectrum to hdf5 format
    with h5py.File('{}.h5'.format(spec_name), 'w') as hf:
        hf.create_dataset('velocity', data=np.array(velocities))
        hf.create_dataset('flux', data=np.array(fluxes))
        hf.create_dataset('wavelength', data=np.array(wavelengths))
        hf.create_dataset('tau', data=np.array(taus))
        hf.create_dataset('noise', data=np.array(noise_vector))
    print('Trident spectrum done [t=%g s]'%(t_elapsed()))


### Select snapshot, line and LOS
s = pg.Snap(snap_file)
ds = yt.load(snap_file)
sim = caesar.load(infile,LoadHalo=False)

cosmo = FlatLambdaCDM(H0=100*sim.simulation.hubble_constant, Om0=sim.simulation.omega_matter, Ob0=sim.simulation.omega_baryon,Tcmb0=2.73)
hubble = cosmo.H(sim.simulation.redshift)
print 'z,H=',sim.simulation.redshift,hubble

# For LOS, default is along z-axis so only x and y coords are needed. Defaults to units of code_length.
print 'logM*,logMHI,SFR=',np.log10(sim.galaxies[igal].masses['stellar']),np.log10(sim.galaxies[igal].masses['HI']),sim.galaxies[igal].sfr
xoffset = 1000.  # in ckpc/h
yoffset = 0.  
los = [xoffset+sim.galaxies[igal].pos[0].value*sim.simulation.hubble_constant, yoffset+sim.galaxies[igal].pos[1].value*sim.simulation.hubble_constant]  # Caesar galaxy igal center
vgal = -sim.galaxies[igal].vel[2].value-(1.e-3*hubble.value*sim.galaxies[igal].pos[2].value)/(1+sim.simulation.redshift)
if igal == -1:
    losx,losy = np.asarray(np.loadtxt('losfile',usecols=(0,1),unpack=True))
    los = np.array([losx[2],losy[2]])
    los = sim.simulation.hubble_constant*(los+0.5)*sim.simulation.boxsize.d
    print 'From losfile, LOS=',los
else:
    print 'LOS=',los,'vgal=',vgal,sim.galaxies[igal].pos[2].value,'dz=',ds.domain_right_edge[2].value * 0.001 * cosmo.H(ds.current_redshift).value / ds.hubble_constant /c
line_tri = 'H I 1216'
line_pg = 'H1215'
#line_pg = 'CIV1548'
lambda_rest = float(pg.analysis.absorption_spectra.lines[line_pg]['l'].split()[0])

generate_pygad_spectrum(s, line_pg, los, 'spectrum_pygad', lambda_rest)
generate_trident_spectrum(ds, line_tri, los, 'spectrum_trident', lambda_rest)

plt.annotate('z=%g'%(np.round(sim.simulation.redshift,1)), xy=(0.9, 0.1), xycoords='axes fraction',size=16,bbox=dict(boxstyle="round", fc="w"),horizontalalignment='right')
plt.axvline(x=vgal,c='k',linestyle=':')   # vertical line at galaxy velocity
plt.legend(loc='best')
plt.show()



