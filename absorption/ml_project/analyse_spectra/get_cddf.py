import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, '/disk04/sapple/cgm/absorption/ml_project/make_spectra/')
from utils import *
from physics import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

cb_blue = '#5289C7'
cb_green = '#90C987'
cb_red = '#E26F72'

cmap = cm.get_cmap('plasma')
gv_colors = [cmap(0.8), cmap(0.6), cmap(0.25)]

def get_bin_middle(xbins):
    return np.array([xbins[i] + 0.5*(xbins[i+1] - xbins[i]) for i in range(len(xbins)-1)])

def quench_thresh(z): # in units of yr^-1 
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask
    
def stop_array_after_inf(array):
    mask = np.isinf(array)
    if len(array[mask]) > 0:
        inf_start = np.where(mask)[0][0]
        array[inf_start:] = np.inf
        return array
    else:
        return array


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    vel_range = 600.
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}1215$', r'${\rm MgII}2796$', r'${\rm CII}1334$',
                  r'${\rm SiIII}1206$', r'${\rm CIV}1548$', r'${\rm OVI}1031$']
    chisq_lim = [4.5, 63.1, 20.0, 70.8, 15.8, 4.5]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    quench = quench_thresh(redshift)

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    inner_outer = [[0.25, 0.5, 0.75], [1.0, 1.25]]
    labels = ['inner', 'outer']
    rho_labels = ['Inner CGM', 'Outer CGM']
    ssfr_labels = ['All', 'Star forming', 'Green valley', 'Quenched']

    logN_min = 11.
    logN_max = 18.
    delta_logN = 0.5
    bins_logN = np.arange(logN_min, logN_max+delta_logN, delta_logN)
    bins_logN = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15., 16., 17., 18.])
    plot_logN = get_bin_middle(bins_logN)
    delta_N = np.array([10**bins_logN[i+1] - 10**bins_logN[i] for i in range(len(plot_logN))])

    idelta = 0.8 / (len(inner_outer) -1)
    icolor = np.arange(0.1, 0.9+idelta, idelta)
    cmap = cm.get_cmap('viridis')
    all_color = [cmap(i) for i in icolor]

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
            
            for j in range(len(fr200)):

                with h5py.File(results_file, 'r') as hf:
                    all_N.extend(hf[f'log_N_{fr200[j]}r200'][:])
                    all_b.extend(hf[f'b_{fr200[j]}r200'][:])
                    all_l.extend(hf[f'l_{fr200[j]}r200'][:])
                    all_ew.extend(hf[f'ew_{fr200[j]}r200'][:])
                    all_chisq.extend(hf[f'chisq_{fr200[j]}r200'][:])
                    all_ids.extend(hf[f'ids_{fr200[j]}r200'][:])

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

            overall_mask = (all_N > logN_min) & (all_N < bins_logN[-1])
            dX = compute_dX(model, wind, snap, lines, len(all_N[overall_mask]), path_lengths)[0]

            plot_data[f'cddf_all'] = np.zeros(len(plot_logN))

            for j in range(len(plot_logN)):
                N_mask = (all_N > logN_min + j*delta_logN) & (all_N < logN_min + (j+1)*delta_logN)
                plot_data[f'cddf_all'][j] = len(all_N[N_mask])
            plot_data[f'cddf_all'] /= (delta_N * dX)
            plot_data[f'cddf_all'] = np.log10(plot_data[f'cddf_all'])

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

                dX_all = compute_dX(model, wind, snap, lines, len(all_ids[overall_mask]), path_lengths)[0]
                dX_sf = compute_dX(model, wind, snap, lines, len(all_ids[sf_mask*overall_mask]), path_lengths)[0]
                dX_gv = compute_dX(model, wind, snap, lines, len(all_ids[gv_mask*overall_mask]), path_lengths)[0]
                dX_q = compute_dX(model, wind, snap, lines, len(all_ids[q_mask*overall_mask]), path_lengths)[0]

                for j in range(len(plot_logN)):
                    N_mask = (all_N > logN_min + j*delta_logN) & (all_N < logN_min + (j+1)*delta_logN)
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
