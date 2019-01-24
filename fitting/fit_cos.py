import h5py
import glob

import sys
sys.path.append('/home/sapple/VAMP')
from vpfits import fit_spectrum

"""
data_dir = '/home/sapple/cgm/cos_samples/pygad/periodic/kpch/'
plot_dir = '/home/sapple/cgm/fitting/plots/'
output = '/home/sapple/cgm/fitting/fits/'
"""

data_dir = '/disk2/sapple/cgm/cos_samples/pygad/periodic/kpch/'
plot_dir = '/disk2/sapple/cgm/fitting/plots/'
output = '/disk2/sapple/cgm/fitting/fits/'

line = 'H1215'

cos_gals = range(44)
cos_gals = [0]
lines = ["H1215" "SiII1260" "CII1335", "SiIII1206", "SiIV1393", "CIII977", "OVI1031"]
direction = ['x_minus', 'x_plus', 'y_minus', 'y_plus']

for cos_gal in cos_gals:
	with h5py.File(data_dir+'samples/cos_galaxy_'+str(cos_gal)+'_sample_data.h5', 'r') as s:
		gal_ids = s['gal_ids'].value
		mass = s['mass'].value
		ssfr = s['ssfr'].value
	s.close()
	for gal in gal_ids:

		ew = []
		N =[]
		N_std = []
		b = []
		b_std = []

		for d in direction:
			filename = data_dir+'spectra/cos_galaxy_'+str(cos_gal)+'_sample_galaxy_'+str(gal)+'_'+d+'.h5'
			with h5py.File(filename, 'r') as f:
				waves = f['wavelength'].value
				#noise = f['noise'].value
				fake_noise = np.array([0.01]*len(waves))
			f.close()
			for line in lines:
				with h5py.File(filename, 'r') as f:
					taus = f[line+'_tau_periodic'].value
				f.close()

				plotname = files[0].split('/', -1)[-1][:-3]
				params, flux_model = fit_spectrum(waves, noise, _tau_periodic, voigt=False, folder=plot_dir+name)
				
				ew.append(params['EW'])
				N.append(parans['N'])
				N_std.append(params['N_std'])
				b.append(params['b'])
				b_std.append(params['b_std'])

		with h5py.File(output+'spectra/cos_galaxy_'+str(cos_gal)+'_sample_galaxy_'+str(gal), 'a') as p:
			p.create_dataset('EW', data=np.array(ew))
			p.create_dataset('N', data=np.array(N))
			p.create_dataset('N_std', data=np.array(N_std))
			p.create_dataset('b', data=np.array(b))
			p.create_dataset('b_std', data=np.array(b_std))
			p.attr['mass'] = mass
			p.attr['ssfr'] = ssfr
			p.attr['gal_id'] = gal
			p.attr['model'] = 'm50n512'
			p.attr['wind'] = 'fh_qr'

		p.close()

