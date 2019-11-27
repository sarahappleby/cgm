import sys
import os
import numpy as np
from astropy.io import ascii
import caesar
import yt
import h5py
import matplotlib.pyplot as plt

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def get_cos_dwarfs():
    table_file = '/home/sapple/cgm/cos_samples/cos_dwarfs/obs_data/line_table_simple.tex'
    table = ascii.read(table_file, format='latex')
    cos_rho = table['Rho']
    cos_M = table['logM_stellar']
    cos_ssfr = table['logsSFR']
    
    # identify galaxies with sSFR lower limit
    ssfr_less_than = np.array([False for i in list(range(len(cos_ssfr)))])
    ssfr_less_than[15] = True
    ssfr_less_than[16] = True
    ssfr_less_than[36:] = np.array([True for i in list(range(7))])
    
    for i, item in enumerate(cos_ssfr):
        if '$<$' in item:
            j = item.find('-')
            cos_ssfr[i] = item[j:]
    cos_ssfr = np.array(cos_ssfr, dtype=float) # SA
    return cos_rho, cos_M, cos_ssfr

model = 'm100n1024'
snap = '151'
wind = 's50j7k'
sample_dir = '/home/sapple/cgm/cos_samples/cos_dwarfs/samples/'
survey = 'dwarfs'
mass_range = 0.1 # dex
ssfr_range = 0.1 # dex # SA
pos_range = 1000. # kpc/h

if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)

if survey == 'dwarfs':
    cos_rho, cos_M, cos_ssfr = get_cos_dwarfs()

print('Loaded COS-Dwarfs survey data')

infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
sim = caesar.load(infile, LoadHalo=False)
gal_cent = np.array([i.central for i in sim.galaxies])

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift

quench = (-1.8+0.3*redshift) - 9. # define galaxy as quenched

gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14) # SA
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession

print('Loaded caesar galaxy data from model ' + model + ' snapshot ' + snap)

choose_mask = np.array([True] * len(sim.galaxies))

gal_ids = np.zeros(43*5)
mass = np.zeros(43*5)
ssfr = np.zeros(43*5)
pos = np.zeros((43*5, 3))
vgal_pos = np.zeros((43*5, 3))

for cos_id in range(43):
        ids = range(cos_id*5, (cos_id+1)*5)
        print('\nFinding the caesar galaxies in the mass and ssfr range of COS Halos galaxy ' + str(cos_id))
        
        mass_range_init = mass_range + 0.
        ssfr_range_init = ssfr_range + 0.
        stop = False
        indices = []
        while not stop:
            mass_mask = (gal_sm >= (cos_M[cos_id] - mass_range_init)) & (gal_sm <= (cos_M[cos_id] + mass_range_init))
            # if cos galaxy is near quenching, we want to match it to galaxies with low sSFR
            if cos_ssfr[cos_id] <= quench:
                ssfr_mask = (gal_ssfr <= (cos_ssfr[cos_id] + ssfr_range_init)) # SA
            else:
                ssfr_mask = (gal_ssfr >= (cos_ssfr[cos_id] - ssfr_range_init)) & (gal_ssfr <= (cos_ssfr[cos_id] + ssfr_range_init)) # SA
            mask = mass_mask * ssfr_mask * gal_cent * choose_mask
            indices = np.where(mask == True)[0]
            
            delete_gals = []
            # check isolation criteria (no other galaxies within 1 Mpc)
            for i, gal in enumerate(indices):
                # compute distance of other galaxies to this one:
                r = np.sqrt(np.sum((gal_pos - gal_pos[gal])**2, axis=1))
                pos_mask = (r.value < pos_range)
                # check for central galaxies in this range
                # one of the galaxies will be the original galaxy, so at least 1 match is expected
                if len(gal_sm[pos_mask]) > 1:
                    delete_gals.append(i)
            if len(delete_gals) > 0.:
                print('Excluding galaxies within 1 Mpc')
                indices = np.delete(indices, delete_gals)

            if len(indices) < 5.: 
                
                if (len(indices) < 2.) & (ssfr_range_init > 5.*ssfr_range) & (mass_range_init > 5.*mass_range):
                    print('No galaxies matching this criteria')
                    stop = True
                    continue

                mass_range_init += 0.05
                ssfr_range_init += 0.05
                print('Expanding sSFR and mass search by 0.05 dex')
                continue
            else:
                stop = True
                continue

        if len(indices) < 1.:
            print ('No galaxies selected')
            continue

	# choose 5 of the galaxies that satisfy the COS-Halos galaxy's conditions
        # choose = np.sort(np.random.choice(range(len(indices)), 5, replace=False))
        # instead, choose the 5 that match most closely in mass

        mass_dev = np.abs(cos_M[cos_id] - gal_sm[indices])
        ssfr_choosing = gal_ssfr[indices]
        if cos_ssfr[cos_id] < quench:
            ssfr_choosing[ssfr_choosing < quench] = cos_ssfr[cos_id]
        ssfr_dev = np.abs(cos_ssfr[cos_id] - ssfr_choosing)
        dev = np.sqrt(mass_dev**2 + ssfr_dev**2)
        choose = np.argsort(dev)[:5]

        print('Chosen galaxies ' + str(indices[choose]))
        print('COS-Dwarfs M*: '+ str(cos_M[cos_id]) + '; selected M* : ' + str(gal_sm[indices[choose]]))
        print('COS-Dwarfs sSFR: ' + str(cos_ssfr[cos_id]) + '; selected sSFR : ' + str(np.array(gal_ssfr[indices[choose]])))

        plt.scatter(cos_M[cos_id], cos_ssfr[cos_id], c='k', marker='x', label='COS-Dwarfs')
        plt.scatter(gal_sm[indices[choose]], np.array(gal_ssfr[indices[choose]]))
        plt.xlim(8, 10.5)
        plt.ylim(-14, -8.5)
        plt.savefig(sample_dir+'plots/cos_id_'+str(cos_id)+'.png')
        plt.clf()

        gal_ids[ids] = indices[choose]
        mass[ids] = gal_sm[indices[choose]]
        ssfr[ids] = gal_ssfr[indices[choose]]
        pos[ids] = gal_pos[indices[choose]]
        vgal_pos[ids] = gal_vgal_pos[indices[choose]]

	# do not repeat galaxies
        choose_mask[indices[choose]] = np.array([False] * 5)

with h5py.File(sample_dir+'/'+model+'_'+wind+'_cos_dwarfs_sample.h5', 'a') as hf:
        hf.create_dataset('gal_ids', data=np.array(gal_ids))
        hf.create_dataset('mass', data=np.array(mass))
        hf.create_dataset('ssfr', data=np.array(ssfr))
        hf.create_dataset('position', data=np.array(pos))
        hf.create_dataset('vgal_position', data=np.array(vgal_pos))
        hf.attrs['pos_units'] = 'kpc/h'
        hf.attrs['mass_units'] = 'Msun'
        hf.attrs['ssfr_units'] = 'Msun/yr'
        hf.attrs['vel_units'] = 'km/s'

plt.scatter(cos_M, cos_ssfr, c='k', marker='x', label='COS-Dwarfs')
plt.scatter(mass, ssfr, s=2., c='b', label='Simba')
plt.xlabel('log M*')
plt.ylabel('log sSFR')
plt.ylim(-14.5, )
plt.legend()
plt.savefig(sample_dir + '/'+model+'_'+wind+'_sample_plot.png')
plt.clf()

