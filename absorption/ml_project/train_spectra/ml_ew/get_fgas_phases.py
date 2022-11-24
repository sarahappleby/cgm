import pandas as pd
import numpy as np
import caesar
from pygadgetreader import readsnap


def ism_phase_line(nH):
    # ISM particles have:
    # log T  = 4 + 1/3 log nH  (+ 1 dex)
    return 5. + 0.33*nH


def get_ism_mask(temp, nH, ism_density):
    nH_mask = nH > ism_density
    ism_line = ism_phase_line(np.log10(nH))
    temp_mask = (np.log10(temp) - ism_line < 0.)
    return temp_mask * nH_mask


if __name__ == '__main__':

    model = 'm100n1024'
    wind = 's50'
    snap = '151'

    photo_temp = 10.**4.5 # in K
    dust_mass_factor = 1.e10
    ism_density = 0.13 # hydrogen number density, cm**-3
    ism_sfr = 0.

    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    h = sim.simulation.hubble_constant
    
    gal_tvir = np.array([i.halo.virial_quantities['temperature'] for i in sim.galaxies])

    snapfile = f'{data_dir}snap_{model}_{snap}.hdf5'
    gas_mass = readsnap(snapfile, 'mass', 'gas', suppress=1, units=1) / h # in Mo
    gas_nh = readsnap(snapfile, 'nh', 'gas', suppress=1, units=1) # in g/cm^3
    gas_delaytime = readsnap(snapfile, 'DelayTime', 'gas', suppress=1)
    gas_temp = readsnap(snapfile, 'u', 'gas', suppress=1, units=1) # in K

    data = pd.read_csv(f'data/{model}_{wind}_{snap}_ew.csv')

    unique_gal_ids = np.unique(data['gal_id'])

    gal_ids_idx = np.array([np.where(unique_gal_ids == i)[0][0] for i in data['gal_id']]).flatten()
   
    fgas_cool = np.zeros(len(unique_gal_ids))
    fgas_warm = np.zeros(len(unique_gal_ids))
    fgas_hot = np.zeros(len(unique_gal_ids))

    for i in range(len(unique_gal_ids)):

        gal = sim.galaxies[unique_gal_ids[i]]

        glist = gal.halo.glist
        ism_gas_mask = get_ism_mask(gas_temp[glist], gas_nh[glist], ism_density)
        cgm_gas_mask = np.invert(ism_gas_mask)
        wind_mask = gas_delaytime[glist] > 0.

        total_gas_mass = np.sum(gas_mass[glist][cgm_gas_mask * np.invert(wind_mask)])
        cool_mask = cgm_gas_mask * np.invert(wind_mask) * (gas_temp[glist] < photo_temp)
        warm_mask = cgm_gas_mask * np.invert(wind_mask) * (gas_temp[glist] > photo_temp) * (gas_temp[glist] < 0.5*gal_tvir[unique_gal_ids[i]])
        hot_mask = cgm_gas_mask * np.invert(wind_mask) * (gas_temp[glist] > 0.5*gal_tvir[unique_gal_ids[i]])
        
        fgas_cool[i] = np.sum(gas_mass[glist][cool_mask]) / total_gas_mass
        fgas_warm[i] = np.sum(gas_mass[glist][warm_mask]) / total_gas_mass
        fgas_hot[i] = np.sum(gas_mass[glist][hot_mask]) / total_gas_mass

    data['fgas_cool'] = fgas_cool[gal_ids_idx]
    data['fgas_warm'] = fgas_warm[gal_ids_idx]
    data['fgas_hot'] = fgas_hot[gal_ids_idx]

    data.to_csv(f'data/{model}_{wind}_{snap}_ew_fgas.csv')
