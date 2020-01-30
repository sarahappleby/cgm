import h5py
import numpy as np
import caesar
import yt

model = 'm50n512'
wind1 = 's50j7k'
wind2 = 's50nojet'
snap = '151'
survey = 'halos'

sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind1+'cos_galaxy_sample.h5'
with h5py.File(sample_file, 'r') as f:
    gal_ids = f['gal_ids'][:]
    cos_ids = f['cos_ids'][:]

match_file = './m50n512/match_halos_'+snap+'.hdf5'
with h5py.File(match_file, 'r') as f:
    prog_index = f[wind1+'_'+wind2][:]

infile = '/home/rad/data/'+model+'/'+wind1+'/Groups/'+model+'_'+snap+'.hdf5'
obj1 = caesar.load(infile, LoadHalo=False)
infile = '/home/rad/data/'+model+'/'+wind2+'/Groups/'+model+'_'+snap+'.hdf5'
obj2 = caesar.load(infile, LoadHalo=False)

new_gal_ids = np.zeros(len(gal_ids))

for i, gal_id in enumerate(gal_ids):
    
    gal = obj1.galaxies[gal_id]
    # for each galaxy in sample, get halo id:
    halo1 = gal.parent_halo_index

    # get new halo id:
    halo2 = obj2.halos[prog_index[halo1]]

    # get new galaxy:
    new_gal_ids[i] = halo2.central_galaxy

# get new galaxy params:

co = yt.utilities.cosmology.Cosmology()
hubble = co.hubble_parameter(obj2.simulation.redshift).in_units('km/s/kpc')
redshift = obj2.simulation.redshift

gal_sm = yt.YTArray([i.masses['stellar'].in_units('Msun') for i in obj2.galaxies], 'Msun')
gal_sfr = yt.YTArray([i.sfr.in_units('Msun/yr') for i in obj2.galaxies], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
gal_pos = yt.YTArray([i.pos.in_units('kpc/h') for i in obj2.galaxies], 'kpc/h')
gal_vels = yt.YTArray([i.vel.in_units('km/s') for i in obj2.galaxies], 'km/s')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in obj2.galaxies ])


new_sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind2+'cos_galaxy_sample.h5'
with h5py.File(new_sample_file, 'a') as f:
    f.create_dataset('gal_ids', data=np.array(new_gal_ids))
    f.create_dataset('cos_ids', data=np.array(cos_ids))
    f.create_dataset('mass', data=np.array(gal_sm[new_gal_ids]))
    f.create_dataset('ssfr', data=np.array(gal_ssfr[new_gal_ids]))
    f.create_dataset('gas_frac', data=np.array(gal_gas_frac[new_gal_ids]))
    f.create_dataset('position', data=np.array(gal_pos[new_gal_ids]))
    f.create_dataset('vgal_position', data=np.array(gal_vgal_pos[new_gal_ids]))
    f.attrs['pos_units'] = 'kpc/h'
    f.attrs['mass_units'] = 'Msun'
    f.attrs['ssfr_units'] = 'Msun/yr'
    f.attrs['vel_units'] = 'km/s'

