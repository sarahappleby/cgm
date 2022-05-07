import numpy as np
import h5py
import pygad as pg
import caesar
from yt.utilities.cosmology import Cosmology
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import create_path_length_file, compute_dX
from cosmic_variance import get_cosmic_variance_cddf

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    vel_range = 600.
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    #chisq_lim = [4.5, 63.1, 20.0, 70.8, 15.8, 4.5] limits with old fitting procedure
    chisq_lim_dict = {'snap_151': [4., 50., 15.8, 39.8, 8.9, 4.5],
                      'snap_137': [3.5, 28.2, 10., 35.5, 8.0, 4.5],
                      'snap_125': [3.5, 31.6, 15.8, 39.8, 10., 5.6], 
                      'snap_105': [4.5, 25.1, 25.1, 34.5, 10., 7.1],}
    chisq_lim = chisq_lim_dict[f'snap_{snap}']

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    boxsize = float(s.boxsize.in_units_of('ckpc/h_0'))
    redshift = s.redshift
    quench = quench_thresh(redshift)

    sim = caesar.load(f'/home/rad/data/{model}/{wind}/Groups/{model}_{snap}.hdf5')
    co = Cosmology(hubble_constant=sim.simulation.hubble_constant, omega_matter=sim.simulation.omega_matter, omega_lambda=sim.simulation.omega_lambda)
    hubble_parameter = co.hubble_parameter(sim.simulation.redshift).in_units('km/s/Mpc')
    hubble_constant = co.hubble_parameter(0).in_units('km/s/Mpc')

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']

    ncells=4
    logN_min = 11.
    logN_max = 18.
    delta_logN = 0.5
    bins_logN = np.arange(logN_min, logN_max+delta_logN, delta_logN)
    bins_logN = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15., 16., 17., 18.])
    plot_logN = get_bin_middle(bins_logN)
    delta_N = np.array([10**bins_logN[i+1] - 10**bins_logN[i] for i in range(len(plot_logN))])

    path_length_file = f'/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/path_lengths.h5'
    if not os.path.isfile(path_length_file):
        create_path_length_file(vel_range, lines, redshift, path_length_file)
    path_lengths = read_h5_into_dict(path_length_file)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/plots/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]
    
    for l, line in enumerate(lines):

        results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
        cddf_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_{line}_cddf_chisqion.h5'

        if os.path.isfile(cddf_file):
            continue

        else:

            plot_data = {}
            plot_data['plot_logN'] = plot_logN.copy()

            all_N = []
            all_b = []
            all_l = []
            all_ew = []
            all_chisq = []
            all_ids = []
            all_los = []
            
            for j in range(len(fr200)):

                with h5py.File(results_file, 'r') as hf:
                    all_N.extend(hf[f'log_N_{fr200[j]}r200'][:])
                    all_b.extend(hf[f'b_{fr200[j]}r200'][:])
                    all_l.extend(hf[f'l_{fr200[j]}r200'][:])
                    all_ew.extend(hf[f'ew_{fr200[j]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{fr200[j]}r200'][:])
                    all_ids.extend(hf[f'ids_{fr200[j]}r200'][:])
                    all_los.extend(hf[f'LOS_pos_{fr200[j]}r200'][:])

            all_N = np.array(all_N)
            all_b = np.array(all_b)
            all_l = np.array(all_l)
            all_ew = np.array(all_ew)
            all_chisq = np.array(all_chisq)
            all_ids = np.array(all_ids)
            all_los = np.array(all_los)

            mask = (all_N > logN_min) * (all_chisq < chisq_lim[l]) * (all_ew >= 0.)
            all_N = all_N[mask]
            all_b = all_b[mask]
            all_l = all_l[mask]
            all_ew = all_ew[mask]
            all_los = all_los[mask]

            overall_mask = (all_N > logN_min) & (all_N < bins_logN[-1])
            dX = compute_dX(len(all_N[overall_mask]), [line], path_lengths, 
                            redshift=redshift, hubble_parameter=hubble_parameter, 
                            hubble_constant=hubble_constant)[0]

            plot_data[f'cddf_all'] = np.zeros(len(plot_logN))

            for j in range(len(bins_logN) -1):
                N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j+1])
                plot_data[f'cddf_all'][j] = len(all_N[N_mask])

            plot_data[f'cddf_all_poisson'] = np.sqrt(plot_data[f'cddf_all'])
            plot_data[f'cddf_all_poisson'] /= (plot_data[f'cddf_all'] * np.log(10.))

            plot_data[f'cddf_all'] /= (delta_N * dX)
            plot_data[f'cddf_all'] = np.log10(plot_data[f'cddf_all'])

            plot_data[f'cddf_all_cv_mean_{ncells}'], plot_data[f'cddf_all_cv_{ncells}'] = \
                    get_cosmic_variance_cddf(all_N, all_los, boxsize, line, bins_logN, delta_N, path_lengths, ncells=ncells, 
                                             redshift=redshift, hubble_parameter=hubble_parameter, hubble_constant=hubble_constant)

            plot_data['cddf_all_err'] = np.sqrt(plot_data[f'cddf_all_poisson']**2. + plot_data[f'cddf_all_cv_{ncells}']**2.)

            for i in range(len(inner_outer)):

                all_N = []
                all_b = []
                all_l = []
                all_ew = []
                all_chisq = []
                all_ids = []

                for j in range(len(inner_outer[i])):
                
                    with h5py.File(results_file, 'r') as hf:
                        all_N.extend(hf[f'log_N_{inner_outer[i][j]}r200'][:])
                        all_b.extend(hf[f'b_{inner_outer[i][j]}r200'][:])
                        all_l.extend(hf[f'l_{inner_outer[i][j]}r200'][:])
                        all_ew.extend(hf[f'ew_{inner_outer[i][j]}r200'][:])
                        all_chisq.extend(hf[f'chisq_{inner_outer[i][j]}r200'][:])
                        all_ids.extend(hf[f'ids_{inner_outer[i][j]}r200'][:])

                all_N = np.array(all_N)
                all_b = np.array(all_b)
                all_l = np.array(all_l)
                all_ew = np.array(all_ew)
                all_chisq = np.array(all_chisq)
                all_ids = np.array(all_ids)

                mask = (all_N > logN_min) * (all_chisq < chisq_lim[l])
                all_N = all_N[mask]
                all_b = all_b[mask]
                all_l = all_l[mask]
                all_ew = all_ew[mask]
                
                all_ids = all_ids[mask]
                idx = np.array([np.where(gal_ids == j)[0] for j in all_ids]).flatten()
                all_mass = mass[idx]
                all_ssfr = ssfr[idx]

                sf_mask, gv_mask, q_mask = ssfr_type_check(quench, all_ssfr)

                plot_data[f'cddf_all_{labels[i]}'] = np.zeros(len(plot_logN))
                plot_data[f'cddf_sf_{labels[i]}'] = np.zeros(len(plot_logN))
                plot_data[f'cddf_gv_{labels[i]}'] = np.zeros(len(plot_logN))
                plot_data[f'cddf_q_{labels[i]}'] = np.zeros(len(plot_logN))

                overall_mask = (all_N > logN_min) & (all_N < bins_logN[-1]) 

                dX_all = compute_dX(len(all_ids[overall_mask]), [line], path_lengths)[0]
                dX_sf = compute_dX(len(all_ids[sf_mask*overall_mask]), [line], path_lengths)[0]
                dX_gv = compute_dX(len(all_ids[gv_mask*overall_mask]), [line], path_lengths)[0]
                dX_q = compute_dX(len(all_ids[q_mask*overall_mask]), [line], path_lengths)[0]

                for j in range(len(bins_logN)-1):
                    N_mask = (all_N > bins_logN[j]) & (all_N < bins_logN[j+1])
                    plot_data[f'cddf_all_{labels[i]}'][j] = len(all_N[N_mask])
                    plot_data[f'cddf_sf_{labels[i]}'][j] = len(all_N[N_mask*sf_mask])
                    plot_data[f'cddf_gv_{labels[i]}'][j] = len(all_N[N_mask*gv_mask])
                    plot_data[f'cddf_q_{labels[i]}'][j] = len(all_N[N_mask*q_mask])

                plot_data[f'cddf_all_{labels[i]}'] /= (delta_N * dX_all)
                plot_data[f'cddf_sf_{labels[i]}'] /= (delta_N * dX_sf)
                plot_data[f'cddf_gv_{labels[i]}'] /= (delta_N * dX_gv)
                plot_data[f'cddf_q_{labels[i]}'] /= (delta_N * dX_q)

                plot_data[f'cddf_all_{labels[i]}'] = np.log10(plot_data[f'cddf_all_{labels[i]}'])
                plot_data[f'cddf_sf_{labels[i]}'] = np.log10(plot_data[f'cddf_sf_{labels[i]}'])
                plot_data[f'cddf_gv_{labels[i]}'] = np.log10(plot_data[f'cddf_gv_{labels[i]}'])
                plot_data[f'cddf_q_{labels[i]}'] = np.log10(plot_data[f'cddf_q_{labels[i]}'])

            write_dict_to_h5(plot_data, cddf_file)
