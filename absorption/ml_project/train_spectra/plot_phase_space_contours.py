import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import numpy as np
import pickle
import pygad as pg
import pandas as pd
import seaborn as sns
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    Nlabels = [r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm MgII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$', 
               r'${\rm log }(N\ {\rm SiIII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CIV} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm OVI} / {\rm cm}^{-2})$']
    x = [0.73, 0.67, 0.7, 0.68, 0.68, 0.69]

    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]
    N_max = 18.

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/train_spectra/plots/'
    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    predictors = ['rho', 'T', 'Z']

    cmap = cm.get_cmap('magma')
    cmap = sns.color_palette("flare_r", as_cmap=True)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')

    i = 0
    j = 0

    for l, line in enumerate(lines):

        data = pd.DataFrame()        

        for p, pred in enumerate(predictors):

            random_forest, features, _, feature_scaler, predictor_scaler, df_full = \
                        pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))

            train = df_full['train_mask']
            data[pred] = df_full[~train][pred]
            data[f'{pred}_pred'] = predictor_scaler.inverse_transform( np.array(random_forest.predict(feature_scaler.transform(df_full[~train][features])).reshape(-1, 1) )).flatten()
        
            if pred == 'rho':
                data[pred] -= np.log10(cosmic_rho)
                data[f'{pred}_pred'] -= np.log10(cosmic_rho)
                data = data.rename(columns={'rho':'delta_rho', 'rho_pred':'delta_rho_pred'})
                data['N'] = df_full[~train]['N']

            if pred == 'Z':
                data[pred] -= np.log10(zsolar[lines.index(line)])
                data[f'{pred}_pred'] -= np.log10(zsolar[lines.index(line)]) 
           
            del df_full

        data.reset_index(drop=True, inplace=True)

        data['delta_error'] = (data['delta_rho'] - data['delta_rho_pred']) / data['delta_rho'] 
        data['T_error'] = (data['T'] - data['T_pred']) / data['T']
        data['error'] = np.sqrt(data['delta_error']**2 + data['T_error']**2)

        g = sns.kdeplot(data=data, x='delta_rho', y='T', ax=ax[i][j], legend=False, cumulative=False, linewidths=1)

        im = ax[i][j].scatter(data['delta_rho_pred'], data['T_pred'], c=np.log10(data['error']), cmap=cmap, s=1, vmin=-2, vmax=1)

        ax[i][j].set_xlim(0, 4)
        ax[i][j].set_ylim(3, 6.1)

        ax[i][j].annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction', 
                          bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
        
        if line in ["SiIII1206", "CIV1548", "OVI1031"]:
            ax[i][j].set_xlabel(r'${\rm log }\delta$')
        
        if line in ['H1215', "SiIII1206"]:
            ax[i][j].set_ylabel(r'${\rm log } (T / {\rm K})$')

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.11, 0.02, 0.77])
    cbar = fig.colorbar(im, cax=cbar_ax, label=r'${\rm log} \sigma$')

    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_lines_deltaT_pred.png', dpi=300)
    plt.close()

