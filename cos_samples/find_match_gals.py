import h5py
import numpy as np
import caesar
import yt

def check_halo_sample(prog_index, obj1, obj2, gal_id):
	gal = obj1.galaxies[gal_id]
	halo1 = gal.parent_halo_index
	halo2 = obj2.halos[prog_index[halo1]]
	return halo2.central_galaxy.GroupID

if __name__ == '__main__':

	model = 'm50n512'
	wind1 = 's50j7k'
	wind_options = ['s50nojet', 's50nox', 's50noagn']
	snap = '137'
	survey = 'halos'

	sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind1+'_cos_'+survey+'_sample.h5'
	with h5py.File(sample_file, 'r') as f:
		gal_ids = np.array(f['gal_ids'][:], dtype='int')
		cos_ids = f['cos_ids'][:]

	infile = '/home/rad/data/'+model+'/'+wind1+'/Groups/'+model+'_'+snap+'.hdf5'
	obj1 = caesar.load(infile)

	for wind2 in wind_options:

		match_file = './m50n512/match_halos_'+snap+'.hdf5'
		with h5py.File(match_file, 'r') as f:
			prog_index = f[wind1+'_'+wind2][:]

		infile = '/home/rad/data/'+model+'/'+wind2+'/Groups/'+model+'_'+snap+'.hdf5'
		obj2 = caesar.load(infile)

		new_gal_ids = np.zeros(len(gal_ids))

		for i, gal_id in enumerate(gal_ids):
			new_gal_ids[i] = check_halo_sample(prog_index, obj1, obj2, gal_id)

		# get new galaxy params:

		new_gal_ids = np.array(new_gal_ids, dtype='int')

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


		new_sample_file = './m50n512/cos_' + survey+'/samples/'+model+'_'+wind2+'_cos_'+survey+'_sample.h5'
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
