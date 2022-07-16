# - get the column densities and overdensities and set the best fit and the running median
#   the best fit is described by a mean and variance
# - for each column density point, predict an overdensity from the best fit and add on 
#   an offset drawn from a Gaussian with the variance of the best fit
# - compare the predictions to the truth.

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
import pygad as pg
import sys
from scipy.optimize import curve_fit

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14.5)

def quench_thresh(z): # in units of yr^-1
    return -1.8  + 0.3*z -9.

def ssfr_type_check(ssfr_thresh, ssfr):

    sf_mask = (ssfr >= ssfr_thresh)
    gv_mask = (ssfr < ssfr_thresh) & (ssfr > ssfr_thresh -1)
    q_mask = ssfr == -14.0
    return sf_mask, gv_mask, q_mask


def bin_data(x, y, xbins):
    digitized = np.digitize(x, xbins)
    return np.array([y[digitized == i] for i in range(1, len(xbins))])


def runningmedian(x, y, xlolim=-1.e20, ylolim=-1.e20, low_per=15.9, high_per=84.1, bins=10):
    
    xp = x[(x>xlolim)&(y>ylolim)]
    yp = y[(x>xlolim)&(y>ylolim)]
    hist,bin_edges=np.histogram(xp,bins)

    ymed = np.zeros(bins)
    yperlo = np.zeros(bins)
    yperhi = np.zeros(bins)
    ndata = np.zeros(bins)
    ystd = np.zeros(bins)

    for i in range(0,len(bin_edges[:-1])):

        ysub = yp[(xp > bin_edges[i]) * (xp < bin_edges[i+1])]

        ymed[i] = np.nanmedian(ysub)
        yperlo[i] = np.nanpercentile(ysub, low_per)
        yperhi[i] = np.nanpercentile(ysub, high_per)
        ndata = len(ysub)
        ystd[i] = np.nanstd(ysub)

    bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])
    
    return bin_edges, bin_cent, ymed, yperlo, yperhi, ystd, ndata


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    line = 'H1215'
    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    x = 0.18
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    galtypes = ['SF', 'GV', 'Q']
    galtypes_long = ['star forming', 'green valley', 'quenched']

    logN = np.arange(12.7, 18, 0.1)
    inner_outer = [[0.25, 0.5], [0.75, 1.0, 1.25]]
    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)
    
    delta_min = 0
    delta_max = 4
    delta_bins = np.arange(delta_min, delta_max+0.2, 0.2)

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)
    quench = quench_thresh(redshift)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/train_spectra/plots_app2022_data/'
    sample_dir = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/'

    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        all_ssfr = sf['ssfr'][:]

    results_file = f'/disk04/sapple/cgm/absorption/ml_project/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'

    N = []
    rho = []
    chisq = []
    ids = []
    all_r = []

    for i in range(len(fr200)):

        with h5py.File(results_file, 'r') as hf:
            N.extend(hf[f'log_N_{fr200[i]}r200'][:])
            rho.extend(hf[f'log_rho_{fr200[i]}r200'][:])
            chisq.extend(hf[f'chisq_{fr200[i]}r200'][:])
            ids.extend(hf[f'ids_{fr200[i]}r200'][:])
            all_r.extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

    N = np.array(N)
    rho = np.array(rho)
    chisq = np.array(chisq)
    ids = np.array(ids)
    all_r = np.array(all_r)

    mask = (N > N_min[lines.index(line)]) * (chisq < chisq_lim[lines.index(line)])
    delta_rho = rho[mask] - np.log10(cosmic_rho)
    N = N[mask]
    all_r = all_r[mask]
    ids = ids[mask]

    idx = np.array([np.where(gal_ids == l)[0] for l in ids]).flatten()
    ssfr = all_ssfr[idx]
    sf_mask, gv_mask, q_mask = ssfr_type_check(quench, ssfr)

    inner_mask = all_r < 0.75
    silly_mask = N < 18.
    fit_mask = N < 15. 

    masks = np.zeros((3, len(fit_mask)))
    masks[0] = sf_mask * fit_mask
    masks[1] = gv_mask * fit_mask
    masks[2] = q_mask * fit_mask

    diff = {galtype: None for galtype in galtypes}
    err = pd.DataFrame(columns=['Galtype', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

    for i in range(3):
        mask = masks[i].astype(bool)
        data = {}

        bin_edges, bin_cent, ymed, yperlo, yperhi, ystd, ndata = runningmedian(N[mask], delta_rho[mask])
        ysiglo = ymed - yperlo
        ysighi = yperhi - ymed
        ysig = (ysiglo + ysighi) * 0.5
        binned_N = bin_data(N[mask], N[mask], bin_edges)
        binned_delta_rho = bin_data(N[mask], delta_rho[mask], bin_edges)
        size = np.sum([len(k) for k in binned_N])
        
        data['delta_rho_pred'] = np.zeros(size)
        data['N'] = np.zeros(size)
        data['delta_rho'] = np.zeros(size)
        
        i_start = 0
        for j in range(len(bin_cent)):
            i_end = i_start + len(binned_N[j])
            

            #lower_gauss = np.random.normal(loc=ymed[j], scale=yperlo[j], size=2*len(binned_N[j]))
            #upper_gauss = np.random.normal(loc=ymed[j], scale=yperhi[j], size=2*len(binned_N[j]))
            #gauss = np.concatenate((lower_gauss[lower_gauss < ymed[j]], upper_gauss[upper_gauss > ymed[j]]))
            #data['delta_rho_pred'][i_start:i_end] = np.random.choice(gauss, size=len(binned_N[j]), replace=False)
            data['delta_rho_pred'][i_start:i_end] = np.random.normal(loc=ymed[j], scale=ystd[j], size=len(binned_N[j]))

            data['N'][i_start:i_end] = binned_N[j]
            data['delta_rho'][i_start:i_end] = binned_delta_rho[j]

            i_start += len(binned_N[j])

        data = pd.DataFrame(data)

        plot_mask = (data['delta_rho'] > delta_min) & (data['delta_rho'] < delta_max)
        
        diff[galtypes[i]] = np.array(data['delta_rho']) - np.array(data['delta_rho_pred'])
        diff_within = round(100* np.sum(diff[galtypes[i]] < 0.2) / len(diff[galtypes[i]]) )

        scores = {}
        scores['Galtype'] = galtypes[i]
        scores['Pearson'] = round(pearsonr(data['delta_rho'],data['delta_rho_pred'])[0],3)
        for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
            scores[_scorer.__name__] = float(_scorer(data['delta_rho'],
                                               data['delta_rho_pred'], multioutput='raw_values'))
        err = err.append(scores, ignore_index=True)

        g = sns.jointplot(data=data[plot_mask], x='delta_rho', y='delta_rho_pred',
                          kind="hex", joint_kws=dict(bins='log', alpha=0.8), xlim=[delta_min, delta_max], ylim=[delta_min, delta_max],
                          marginal_ticks=True, marginal_kws=dict(bins=delta_bins, fill=False, stat='probability'))

        g.figure.axes[0].plot(delta_bins, delta_bins, ls=':', lw=2, c='k')
        g.set_axis_labels(xlabel=r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm True}$', 
                          ylabel=r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm Pred}$')

        g.figure.axes[1].set_yticks([0.1])
        g.figure.axes[2].set_xticks([0.1])

        g.figure.axes[0].text(0.57, 0.05, 'Predictions within\n 0.2 dex: {}\%'.format(diff_within),
                              transform=g.figure.axes[0].transAxes)
        g.figure.axes[0].text(0.29, 0.93, 'Running median, {}'.format(galtypes_long[i]),
                              transform=g.figure.axes[0].transAxes)

        cax = g.figure.add_axes([x, .6, .02, .2])
        g.figure.colorbar(mpl.cm.ScalarMappable(norm=g.figure.axes[0].collections[0].norm, cmap=g.figure.axes[0].collections[0].cmap),
                          cax=cax, label=r'$n$')

        plt.savefig(f'{plot_dir}/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_runmed_{galtypes[i]}.png')
        plt.close()

