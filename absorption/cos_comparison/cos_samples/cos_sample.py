# This script selects galaxies from Simba that match conditions in either the COS-Dwarfs or COS-Halos survey.


import sys
import os
import numpy as np
import caesar
import yt
import h5py
import matplotlib.pyplot as plt

def delete_indices(indices, delete_gals):
    if len(delete_gals) > 0.:
        indices = np.delete(indices, delete_gals)
    return indices

def get_halo_id(obj, gal_id):
    gal = obj.galaxies[gal_id]
    return gal.parent_halo_index

def check_prog_sample(prog_index, obj1, obj2, gal_id):
    halo1_id = get_halo_id(obj1, gal_id)
    halo2 = obj2.halos[prog_index[halo1_id]]

    return int(halo2.central_galaxy.GroupID)

def check_r200_sample(r200_file, halo_id, wind):
    # returns a True if the r200 is zero
    with h5py.File(r200_file, 'r') as f:
        r200_all = f[wind+'_halo_r200'][:]
    r200 = r200_all[halo_id]
    
    return (r200 == 0.)

def halo_check(sim, objs, prog_index, indices, r200_file):
    wind_option = ['s50nojet', 's50nox', 's50noagn'] 

    delete_gals = []
    halo_ids = np.array([get_halo_id(sim, i) for i in indices])
    # get that the fiducial box halo has a non-zero r200
    check_array = check_r200_sample(r200_file, halo_ids, 's50j7k')
    if True in check_array:
        delete_gals.append(np.arange(len(indices))[check_array])
    
    # for each halo, check it in the other wind boxes and check r200 is not zero
    for j, w in enumerate(wind_options):
        w_halo_ids = np.zeros(len(indices))
        check_array = np.array([False] * len(indices))
        for i, gal in enumerate(indices):
            try:
                w_halo_ids[i] = check_prog_sample(prog_index[j], sim, objs[j], gal)
            except AttributeError:
                check_array[i] = True
                w_halo_ids[i] = np.nan
                continue
    
        check_array[~np.isnan(w_halo_ids)] = check_r200_sample(r200_file, w_halo_ids[~np.isnan(w_halo_ids)].astype('int'), w)
        if True in check_array:
            delete_gals.append(np.arange(len(indices))[check_array])

    delete_gals = [item for sublist in delete_gals for item in sublist]
    delete_gals = np.unique(np.array(delete_gals))

    print('Excluding galaxies with no central counterpart in other wind boxes')
    indices = delete_indices(indices, delete_gals)

    return indices

def isolation_check(gal_pos, pos_range, gal_cent, indices):
    
    delete_gals = []
    # check isolation criteria (no other galaxies within 1 Mpc)
    for i, gal in enumerate(indices):
        # compute distance of other galaxies to this one:
        r = np.sqrt(np.sum((gal_pos - gal_pos[gal])**2, axis=1))
        pos_mask = (r.value < pos_range) * gal_cent
        # check for central galaxies in this range
        # one of the galaxies will be the original galaxy, so at least 1 match is expected
        if len(gal_pos[pos_mask]) > 1:
            delete_gals.append(i)

    print('Excluding galaxies within 1 Mpc')
    indices = delete_indices(indices, delete_gals)
    
    return indices

# for COS-Halos, run at snap='137' and survey = 'halos'
# for COS-Dwarfs, run at snap='151' and survey = 'dwarfs'

if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    survey = sys.argv[1]

    sample_dir = '/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/'+model+'/cos_'+survey+'/samples/'
    mass_range = 0.1 # dex
    mass_range_lim = 0.15 # limit of how far away in mass dex we can look
    ssfr_range = 0.1 # dex
    ssfr_range_lim = 0.25 # limit of how far away in ssfr dex we can look (excludes quenched galaxies)
    pos_range = 1000. # kpc/h
    mlim = np.log10(5.8e8) # lower limit of M*
    ngals_each = 5
    
    # set to True if we want to have the isolation criteria
    do_isolation = True
    # set to True if we want to check for halos in other wind boxes
    do_halo_check = False
    if do_halo_check: wind_options = ['s50nojet', 's50nox', 's50noagn']

    if not os.path.exists(sample_dir):
    	os.makedirs(sample_dir)

    # load in cos survey data and get rid of low mass objects
    if survey == 'dwarfs':
        from get_cos_info import get_cos_dwarfs
        cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_dwarfs()
        snap = '151'
    elif survey == 'halos':
        from get_cos_info import get_cos_halos
        cos_rho, cos_M, cos_r200, cos_ssfr = get_cos_halos()
        snap = '137'

    cos_ids = np.arange(len(cos_M))[cos_M > mlim]
    cos_rho = cos_rho[cos_M > mlim]
    cos_ssfr = cos_ssfr[cos_M > mlim]
    cos_r200 = cos_r200[cos_M > mlim]
    cos_M = cos_M[cos_M > mlim]
    numgals = len(cos_M)
    print('Loaded COS-Dwarfs survey data')

    infile = '/home/rad/data/'+model+'/'+wind+'/Groups/'+model+'_'+snap+'.hdf5'
    sim = caesar.load(infile)
    gal_cent = np.array([i.central for i in sim.galaxies])
    co = yt.utilities.cosmology.Cosmology()
    hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
    redshift = sim.simulation.redshift
    quench = (-1.8+0.3*redshift) - 9. # define galaxy as quenched

    gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
    gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
    gal_ssfr = gal_sfr / gal_sm
    gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
    gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
    gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
    gal_sm = np.log10(gal_sm)
    gal_recession = gal_pos.in_units('kpc')*hubble
    gal_vgal_pos = gal_vels + gal_recession
    gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

    print('Loaded caesar galaxy data from model ' + model + ' snapshot ' + snap)

    # load in the other wind boxes if we need them
    if do_halo_check:
        objs = []
        prog_index = []
        match_file = './m50n512/match_halos_'+snap+'.hdf5'
        E
        r200_file = '/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/m50n512/m50n512_'+snap+'_r200_info.h5'
        print('Loading other wind snaps for halo check')
        for w in wind_options:
            infile_new = '/home/rad/data/'+model+'/'+w+'/Groups/'+model+'_'+snap+'.hdf5'
            objs.append(caesar.load(infile_new))
            with h5py.File(match_file, 'r') as f:
                prog_index.append(f[wind+'_'+w][:])

    # initially we can choose any galaxy
    choose_mask = np.array([True] * len(sim.galaxies))

    # empty arrays to store 5 simba galaxies per cos galaxy
    gal_ids = np.ones(numgals*ngals_each) * np.nan
    mass = np.ones(numgals*ngals_each) * np.nan
    ssfr = np.ones(numgals*ngals_each) * np.nan
    gas_frac = np.ones(numgals*ngals_each) * np.nan
    pos = np.ones((numgals*ngals_each, 3)) * np.nan
    vgal_pos = np.ones((numgals*ngals_each, 3)) * np.nan
    halo_pos = np.ones((numgals*ngals_each, 3)) * np.nan
    halo_r200 = np.ones((numgals*ngals_each)) * np.nan

    for cos_id in np.flip(np.argsort(cos_M)):
            ids = range(cos_id*ngals_each, (cos_id+1)*ngals_each)
            print('\nFinding the caesar galaxies in the mass and ssfr range of COS Halos galaxy ' + str(cos_id))
            
            mass_range_init = mass_range + 0.
            ssfr_range_init = ssfr_range + 0.
            stop = False
            indices = []
            while not stop:
                # get galaxy mask
                mass_mask = (gal_sm >= (cos_M[cos_id] - mass_range_init)) & (gal_sm <= (cos_M[cos_id] + mass_range_init)) & (gal_sm > mlim)
                # if cos galaxy is near quenching, we want to match it to galaxies with low sSFR
                if cos_ssfr[cos_id] <= quench:
                    ssfr_mask = (gal_ssfr <= (cos_ssfr[cos_id] + ssfr_range_init))
                else:
                    ssfr_mask = (gal_ssfr >= (cos_ssfr[cos_id] - ssfr_range_init)) & (gal_ssfr <= (cos_ssfr[cos_id] + ssfr_range_init))
                mask = mass_mask * ssfr_mask * gal_cent * choose_mask
                indices = np.where(mask == True)[0]
                
                if do_isolation:
                    indices = isolation_check(gal_pos, pos_range, gal_cent, indices)

                if do_halo_check & len(indices) > 0:
                    indices = halo_check(sim, objs, prog_index, indices, r200_file)
                if not do_halo_check:
                    _r200 = np.array([i.halo.virial_quantities['r200c'].in_units('kpc/h') for i in np.array(sim.galaxies)[indices]])
                    delete_gals = np.where(_r200 == 0.)[0]
                    indices = delete_indices(indices, delete_gals)

                if len(indices) < ngals_each:                     
                    if (ssfr_range_init > ssfr_range_lim) or (mass_range_init > mass_range_lim):
                        print('Insufficient galaxies matching this criteria')
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

            # choose the ngals_each (5) that match most closely in mass and ssfr
            mass_dev = np.abs(cos_M[cos_id] - gal_sm[indices])
            ssfr_choosing = gal_ssfr[indices]
            if cos_ssfr[cos_id] < quench:
                ssfr_choosing[ssfr_choosing < quench] = cos_ssfr[cos_id]
            ssfr_dev = np.abs(cos_ssfr[cos_id] - ssfr_choosing)
            dev = np.sqrt(mass_dev**2 + ssfr_dev**2)
            choose = np.argsort(dev)[:ngals_each]

            print('Chosen galaxies ' + str(indices[choose]))
            print('COS-Dwarfs M*: '+ str(cos_M[cos_id]) + '; selected M* : ' + str(gal_sm[indices[choose]]))
            print('COS-Dwarfs sSFR: ' + str(cos_ssfr[cos_id]) + '; selected sSFR : ' + str(np.array(gal_ssfr[indices[choose]])))

            plt.scatter(cos_M[cos_id], cos_ssfr[cos_id], c='k', marker='x', label='COS-Dwarfs')
            plt.scatter(gal_sm[indices[choose]], np.array(gal_ssfr[indices[choose]]))
            if survey == 'dwarfs':
                plt.xlim(8., 10.5)
            elif survey == 'halos':
                plt.xlim(9.5, 12.)
            plt.ylim(-14, -8.5)
            plt.savefig(sample_dir+'plots/cos_id_'+str(cos_id)+'.png')
            plt.clf()

            # in case we have fewer galaxies than required, lets fill gaps with nans:
            if ngals_each - len(indices[choose]) > 0.:
                empty = np.ones(ngals_each - len(indices)) * np.nan

                gal_ids[ids] = np.concatenate((indices[choose], empty))
                mass[ids] = np.concatenate((gal_sm[indices[choose]], empty))
                ssfr[ids] = np.concatenate((gal_ssfr[indices[choose]], empty))
                gas_frac[ids] = np.concatenate((gal_gas_frac[indices[choose]], empty))
                pos[ids] = np.concatenate((gal_pos[indices[choose]], np.transpose([empty for _ in range(3)])))
                vgal_pos[ids] = np.concatenate((gal_vgal_pos[indices[choose]], np.transpose([empty for _ in range(3)])))
            else:
                gal_ids[ids] = indices[choose]
                mass[ids] = gal_sm[indices[choose]]
                ssfr[ids] = gal_ssfr[indices[choose]]
                gas_frac[ids] = gal_gas_frac[indices[choose]]
                pos[ids] = gal_pos[indices[choose]]
                vgal_pos[ids] = gal_vgal_pos[indices[choose]]

    	        # do not repeat galaxies
                choose_mask[indices[choose]] = np.array([False] * (len(indices[choose])))
    
    halo_r200[~np.isnan(gal_ids)] = \
            np.array([sim.galaxies[int(i)].halo.virial_quantities['r200c'].in_units('kpc/h') for i in gal_ids if ~np.isnan(i)])
    halo_pos[~np.isnan(gal_ids)] = \
            np.array([sim.galaxies[int(i)].halo.pos.in_units('kpc/h') for i in gal_ids if ~np.isnan(i)])

    with h5py.File(sample_dir+'/'+model+'_'+wind+'_cos_'+survey+'_sample.h5', 'a') as hf:
            hf.create_dataset('cos_ids', data=np.array(cos_ids))
            hf.create_dataset('gal_ids', data=np.array(gal_ids))
            hf.create_dataset('mass', data=np.array(mass))
            hf.create_dataset('ssfr', data=np.array(ssfr))
            hf.create_dataset('gas_frac', data=np.array(gas_frac))
            hf.create_dataset('position', data=np.array(pos))
            hf.create_dataset('halo_r200', data=np.array(halo_r200))
            hf.create_dataset('halo_pos', data=np.array(halo_pos))
            hf.create_dataset('vgal_position', data=np.array(vgal_pos))
            hf.attrs['pos_units'] = 'kpc/h'
            hf.attrs['mass_units'] = 'Msun'
            hf.attrs['ssfr_units'] = 'Msun/yr'
            hf.attrs['vel_units'] = 'km/s'

    plt.scatter(cos_M, cos_ssfr, c='k', marker='x', label='COS-'+survey)
    plt.scatter(mass, ssfr, s=2., c='b', label='Simba')
    plt.xlabel('log M*')
    plt.ylabel('log sSFR')
    plt.ylim(-14.5, )
    plt.legend()
    plt.savefig(sample_dir + '/'+model+'_'+wind+'_sample_plot.png')
    plt.clf()

