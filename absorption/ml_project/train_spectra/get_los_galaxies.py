import numpy as np
import h5py
import sys
import yt
import caesar

def handle_periodic_boundaries(position, boxsize):
    position[position < 0] += boxsize
    position[position > boxsize] -= boxsize
    return position

def handle_periodic_mask(sample_gal_vpos, gal_vpos, vel_range, vel_boxsize):
    mask = (gal_vpos > sample_gal_vpos - vel_range) & (gal_vpos < sample_gal_vpos + vel_range)
    dv_right_edge = vel_boxsize - sample_gal_vpos
    if dv_right_edge < vel_range: 
        mask = mask | (gal_vpos < vel_range - dv_right_edge)
    return mask

def move_edge_galaxies(los, gal_pos, rho, boxsize):
    for axis in range(2):
        if los[axis] < rho:
            right_gals = gal_pos[:, axis] > (boxsize - rho)
            gal_pos[:, axis][right_gals] -= boxsize
        elif los[axis] > (boxsize - rho):
            left_gals = gal_pos[:, axis] < rho
            gal_pos[:, axis][left_gals] += boxsize
    return gal_pos

def los_galaxies(los, gal_pos, rho, boxsize, vel_mask):
    new_gal_pos = move_edge_galaxies(los, gal_pos, rho, boxsize)
    dx = new_gal_pos[:, 0] - los[0]
    dy = new_gal_pos[:, 1] - los[1]
    dr = np.sqrt(dx**2 + dy**2)
    los_mask = dr < rho
    return np.arange(len(gal_pos))[los_mask*vel_mask]


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    sqrt2 = np.sqrt(2.)
    vel_range = 600. # km/s
    vel_boxsize = 10000. #km/s
    boxsize = 100000. # kpc/h
    orients = [0, 45, 90, 135, 180, 225, 270, 315]
    
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'
    data_dir = f'/home/rad/data/{model}/{wind}/'
    sim =  caesar.load(f'{data_dir}Groups/{model}_{snap}.hdf5')
    co = yt.utilities.cosmology.Cosmology(hubble_constant=sim.simulation.hubble_constant,
                                          omega_matter=sim.simulation.omega_matter,
                                          omega_lambda=sim.simulation.omega_lambda)
    hubble = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/kpc')
    redshift = sim.simulation.redshift

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
    gal_vpos = gal_vels + gal_recession
    gal_vpos = np.array(gal_vpos[:, 2])
    gal_gas_frac = np.array([i.masses['gas'].in_units('Msun') /i.masses['stellar'].in_units('Msun') for i in sim.galaxies ])

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        sample_gal_ids = sf['gal_ids'][:]
        sample_gal_pos = (sf['position'][:]  *  (1.+redshift))
        sample_gal_vpos = sf['vgal_position'][:][:, 2]
        sample_gal_r200 = sf['halo_r200'][:] * (1.+redshift) # already in kpc/h, factor of 1+z for comoving

    gal_pos = np.array(gal_pos)
    # handle periodic boundaries
    gal_vpos = handle_periodic_boundaries(gal_vpos, vel_boxsize)
    sample_gal_vpos = handle_periodic_boundaries(sample_gal_vpos, vel_boxsize)

    gal_ids_long = []
    associated_gal_ids = []
    all_r200 = []
    all_orients = []

    for i, gal in enumerate(sample_gal_ids):

        for j in range(len(fr200)):

            rho = sample_gal_r200[i] * fr200[j]
            vel_mask = (gal_vpos > sample_gal_vpos[i] - vel_range) & (gal_vpos < sample_gal_vpos[i] + vel_range) 
            # handle periodic boundaries
            dv_right_edge = vel_boxsize - sample_gal_vpos[i]
            if dv_right_edge < vel_range:
                vel_mask = vel_mask | (gal_vpos  < vel_range - dv_right_edge)
   
            los = np.tile(sample_gal_pos[i][:2], len(orients)).reshape(len(orients), 2)
            los[0][0] += rho # orient 0 degrees
            los[1][0] += (rho / sqrt2); los[1][1] += (rho / sqrt2) # orient 45 degrees 
            los[2][1] += rho # orient 90 degrees
            los[3][0] -= (rho / sqrt2); los[3][1] += (rho / sqrt2) # orient 135 degrees
            los[4][0] -= rho # orient 180 degrees
            los[5][0] -= (rho / sqrt2); los[5][1] -= (rho / sqrt2) # orient 225 degrees
            los[6][1] -= rho # orient 270 degrees
            los[7][0] += (rho / sqrt2); los[7][1] -= (rho / sqrt2) # orient 315 degrees
    
            for k in range(len(los)):
                ids = los_galaxies(los[k], gal_pos, rho, boxsize, vel_mask)
                if len(ids > 0):
                    ids = np.delete(ids, np.where(ids == gal)[0])
                if len(ids > 0):
                    gal_ids_long.extend(np.repeat(gal, len(ids)))
                    associated_gal_ids.extend(ids)
                    all_r200.extend(np.repeat(fr200[j], len(ids)))
                    all_orients.extend(np.repeat(orients[k], len(ids)))

    gal_ids_long = np.array(gal_ids_long)
    associated_gal_ids = np.array(associated_gal_ids)
    all_r200 = np.array(all_r200)
    all_orients = np.array(all_orients)

