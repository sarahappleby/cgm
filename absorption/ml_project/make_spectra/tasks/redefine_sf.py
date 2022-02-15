import caesar
import yt
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rcParams["figure.figsize"] = (7,6)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9. 


def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


model = sys.argv[1]
wind = sys.argv[2]
snap = sys.argv[3]

sf_cmap = plt.get_cmap('jet_r')
sf_cmap = truncate_colormap(sf_cmap, 0.1, 0.9)

delta_m = 0.25
min_m = 10.
nbins_m = 6
mass_bins = np.arange(min_m, min_m+(nbins_m+1)*delta_m, delta_m)
ngals_each = 12
nbins_ssfr = 3

sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
data_dir = f'/home/rad/data/{model}/{wind}/'
sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
co = yt.utilities.cosmology.Cosmology(hubble_constant=sim.simulation.hubble_constant,
                                      omega_matter=sim.simulation.omega_matter,
                                      omega_lambda=sim.simulation.omega_lambda)
hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
redshift = sim.simulation.redshift
quench = quench_thresh(redshift)

gal_cent = np.array([i.central for i in sim.galaxies])
gal_sm = yt.YTArray([sim.galaxies[i].masses['stellar'].in_units('Msun') for i in range(len(sim.galaxies))], 'Msun')
gal_sfr = yt.YTArray([sim.galaxies[i].sfr.in_units('Msun/yr') for i in range(len(sim.galaxies))], 'Msun/yr')
gal_ssfr = gal_sfr / gal_sm
gal_ssfr = np.log10(gal_ssfr.value + 1e-14)
gal_pos = yt.YTArray([sim.galaxies[i].pos.in_units('kpc/h') for i in range(len(sim.galaxies))], 'kpc/h')
gal_vels = yt.YTArray([sim.galaxies[i].vel.in_units('km/s') for i in range(len(sim.galaxies))], 'km/s')
gal_lgas = yt.YTArray([sim.galaxies[i].rotation['gas_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_lbaryon = yt.YTArray([sim.galaxies[i].rotation['baryon_L'].in_units('Msun*km*kpc/(h*s)') for i in range(len(sim.galaxies))], 'Msun*km*kpc/(h*s)')
gal_sm = np.log10(gal_sm)
gal_recession = gal_pos.in_units('kpc')*hubble
gal_vgal_pos = gal_vels + gal_recession
gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
    sample_ids = sf['gal_ids'][:]
    sample_ssfr = sf['ssfr'][:]
    sample_sm = sf['mass'][:]

new_gal_ids = np.ones(nbins_m*nbins_ssfr*ngals_each) * np.nan

sf_mask_sample, gv_mask_sample, q_mask_sample = ssfr_type_check(quench, sample_ssfr)
sf_mask, gv_mask, q_mask = ssfr_type_check(quench, gal_ssfr)

for i in range(nbins_m):
    gal_id_range = range((i)*ngals_each*nbins_ssfr, (i+1)*ngals_each*nbins_ssfr)
    mass_mask_sample = (sample_sm > mass_bins[i]) & (sample_sm < (mass_bins[i] + delta_m))
    mass_mask = (gal_sm > mass_bins[i]) & (gal_sm < (mass_bins[i] + delta_m))

    # get the star forming galaxies:
    mask_sample = mass_mask_sample * sf_mask_sample
    mask = gal_cent * mass_mask * sf_mask
    if len(sample_ids[mask_sample]) == ngals_each:
        new_gal_ids[gal_id_range[0:ngals_each]] = sample_ids[mask_sample]
    
    elif len(sample_ids[mask_sample]) < ngals_each:
        new_gal_ids[gal_id_range[0:len(sample_ids[mask_sample])]] = sample_ids[mask_sample]
        shortage = ngals_each - len(sample_ids[mask_sample])
        remove = np.in1d(np.arange(len(gal_sm))[mask], sample_ids[mask_sample]).nonzero()[0]
        available = np.delete(np.arange(len(gal_sm))[mask], remove)
        new_gal_ids[gal_id_range[len(sample_ids[mask_sample]):ngals_each]] =  np.random.choice(available, size=shortage, replace=False)
    
    elif len(sample_ids[mask_sample]) > ngals_each:
        new_gal_ids[gal_id_range[0:ngals_each]] = np.random.choice(sample_ids[mask_sample], size=ngals_each, replace=False)

    # get the green valley galaxies:
    mask_sample = mass_mask_sample * gv_mask_sample
    mask = gal_cent * mass_mask * gv_mask
    if len(sample_ids[mask_sample]) == ngals_each:
        new_gal_ids[gal_id_range[ngals_each:2*ngals_each]] = sample_ids[mask_sample]
    
    elif len(sample_ids[mask_sample]) < ngals_each:
        new_gal_ids[gal_id_range[ngals_each:ngals_each+len(sample_ids[mask_sample])]] = sample_ids[mask_sample]
        shortage = ngals_each - len(sample_ids[mask_sample])
        remove = np.in1d(np.arange(len(gal_sm))[mask], sample_ids[mask_sample]).nonzero()[0]
        available = np.delete(np.arange(len(gal_sm))[mask], remove)
        new_gal_ids[gal_id_range[ngals_each+len(sample_ids[mask_sample]):2*ngals_each]] =  np.random.choice(available, size=shortage, replace=False)
    
    elif len(sample_ids[mask_sample]) > ngals_each:
        new_gal_ids[gal_id_range[ngals_each:2*ngals_each]] = np.random.choice(sample_ids[mask_sample], size=ngals_each, replace=False)

    # get the quenched galaxies:
    mask_sample = mass_mask_sample * q_mask_sample
    mask = gal_cent * mass_mask * q_mask
    if len(sample_ids[mask_sample]) == ngals_each:
        new_gal_ids[gal_id_range[2*ngals_each:3*ngals_each]] = sample_ids[mask_sample]

    elif len(sample_ids[mask_sample]) < ngals_each:
        new_gal_ids[gal_id_range[2*ngals_each:2*ngals_each+len(sample_ids[mask_sample])]] = sample_ids[mask_sample]
        shortage = ngals_each - len(sample_ids[mask_sample])
        remove = np.in1d(np.arange(len(gal_sm))[mask], sample_ids[mask_sample]).nonzero()[0]
        available = np.delete(np.arange(len(gal_sm))[mask], remove)
        new_gal_ids[gal_id_range[2*ngals_each+len(sample_ids[mask_sample]):3*ngals_each]] =  np.random.choice(available, size=shortage, replace=False)
    
    elif len(sample_ids[mask_sample]) > ngals_each:
        new_gal_ids[gal_id_range[2*ngals_each:3*ngals_each]] = np.random.choice(sample_ids[mask_sample], size=ngals_each, replace=False)

new_gal_ids = new_gal_ids.astype('int')

# replace repeated ids from original sample
u, c = np.unique(new_gal_ids, return_counts=True)
dups = u[c > 1].astype('int')

if len(dups) > 0:

    mass_bin_edges = mass_bins[np.digitize(gal_sm[dups],mass_bins) - 1] # left edges of the relevant mass bins
    available = np.array([True for i in range(len(gal_sm))])
    available[new_gal_ids] = False

    for i in range(len(dups)):
   
        replace = np.where(new_gal_ids == dups[i])[0][1:]

        mass_mask = (gal_sm > mass_bin_edges[i]) & (gal_sm < mass_bin_edges[i]+delta_m)
        if gal_ssfr[dups[i]] > quench:
            ssfr_mask = gal_ssfr > quench
        elif (gal_ssfr[dups[i]] < quench) & (gal_ssfr[dups[i]] > (quench - 1.)):
            ssfr_mask = (gal_ssfr > (quench -1.)) & (gal_ssfr < quench)
        else:
            ssfr_mask = (gal_ssfr == -14.)
    
        for j in replace:
            mask = mass_mask * gal_cent * ssfr_mask * available
            ids = np.arange(len(gal_sm))[mask]
            try:
                new_gal_ids[j] = np.random.choice(ids, size=1, replace=False)
                available[new_gal_ids[j]] = False
            except ValueError:
                continue

new_gal_ids = np.delete(new_gal_ids, np.where(new_gal_ids == -99)[0])

halo_r200 = np.array([sim.galaxies[i].halo.virial_quantities['r200c'].in_units('kpc/h') for i in new_gal_ids])
halo_pos = np.array([sim.galaxies[i].halo.pos.in_units('kpc/h') for i in new_gal_ids])

with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'a') as hf:
    hf.create_dataset('gal_ids', data=np.array(new_gal_ids))
    hf.create_dataset('mass', data=np.array(gal_sm[new_gal_ids]))
    hf.create_dataset('ssfr', data=np.array(gal_ssfr[new_gal_ids]))
    hf.create_dataset('sfr', data=np.array(gal_sfr[new_gal_ids]))
    hf.create_dataset('fgas', data=np.array(gal_gas_frac[new_gal_ids]))
    hf.create_dataset('position', data=np.array(gal_pos[new_gal_ids]))
    hf.create_dataset('vgal_position', data=np.array(gal_vgal_pos[new_gal_ids]))
    hf.create_dataset('L_gas', data=np.array(gal_lgas[new_gal_ids]))
    hf.create_dataset('L_baryon', data=np.array(gal_lbaryon[new_gal_ids]))
    hf.create_dataset('halo_r200', data=np.array(halo_r200))
    hf.create_dataset('halo_pos', data=np.array(halo_pos))
    hf.attrs['pos_units'] = 'kpc/h'
    hf.attrs['mass_units'] = 'log Msun'
    hf.attrs['ssfr_units'] = 'log yr^-1'
    hf.attrs['sfr_units'] = 'log Msun/yr'
    hf.attrs['vel_units'] = 'km/s'

