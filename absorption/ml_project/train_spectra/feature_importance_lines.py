import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import pandas as pd
import pygad as pg
import pickle
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

from scipy.stats import pearsonr

np.random.seed(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = sys.argv[4]

    features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot'] 
    predictors = ['rho', 'T', 'Z']

    features_pretty = [r'${\rm log} N$', r'$b$', r'${\rm log\ EW}$', 
                       r'${\rm d}v$', r'$f_{r200}$', r'${\rm log} M_\star$',
                       r'${\rm sSFR}$', r'$\kappa_{\rm rot}$']

    predictors_pretty = [r'${\rm log}\ \delta$', r'${\rm log}\ T$', r'${\rm log}\ Z$']

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    #cmap = matplotlib.cm.viridis
    #cmap.set_bad(color='white')

    cmap = sns.color_palette("flare_r", as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 9))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.295, 0.02, 0.4])


    importance = np.zeros((3, len(features), len(features)))

    for p, pred in enumerate(predictors):

        gridsearch, _, _, _, _, df_full = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))
        train = df_full['train_mask']

        err = pd.DataFrame(columns=['Feature removed', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

        for i in range(len(features)):
            
            features_use = np.delete(features, i)
            idx = np.delete(np.arange(len(features)), i)

            feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features_use])
            predictor_scaler = preprocessing.StandardScaler().fit(np.array(df_full[train][pred]).reshape(-1, 1) )

            random_forest = RandomForestRegressor(n_estimators=gridsearch.best_params_['n_estimators'],
                                                  min_samples_split=gridsearch.best_params_['min_samples_split'],
                                                  min_samples_leaf=gridsearch.best_params_['min_samples_leaf'],)
            random_forest.fit(feature_scaler.transform(df_full[train][features_use]), predictor_scaler.transform(np.array(df_full[train][pred]).reshape(-1, 1) ))

            importance[p][i][idx] = random_forest.feature_importances_

            conditions_pred = predictor_scaler.inverse_transform(np.array( random_forest.predict(feature_scaler.transform(df_full[~train][features_use]))).reshape(-1, 1) )
            conditions_true = pd.DataFrame(df_full[~train],columns=[pred]).values

            conditions_pred = conditions_pred.flatten()
            conditions_true = conditions_true.flatten()

            if pred == 'rho':
                conditions_pred -= np.log10(cosmic_rho)
                conditions_true -= np.log10(cosmic_rho)

            if pred == 'Z':
                conditions_pred -= np.log10(zsolar[lines.index(line)])
                conditions_true -= np.log10(zsolar[lines.index(line)])

            scores = {}
            scores['Feature removed'] = features[i]
            scores['Pearson'] = round(pearsonr(conditions_true, conditions_pred)[0],3)
            for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
                scores[_scorer.__name__] = float(_scorer(conditions_pred,
                                                         conditions_true, multioutput='raw_values'))
            err = err.append(scores, ignore_index=True)

        print(pred, err)

        ### plot importance matrix

        importance_use = np.transpose(importance[p])
        mask = importance_use == 0

        if p == len(predictors) - 1:
            g = sns.heatmap(importance_use, mask=mask, cmap=cmap, vmax=1, vmin=0, annot=False, ax=ax[p], square=True, linewidths=.5, 
                            cbar_ax=cbar_ax, cbar_kws={'label': 'Importance'})
        else:
            g = sns.heatmap(importance_use, mask=mask, cmap=cmap, vmax=1, vmin=0, annot=False, ax=ax[p], square=True, linewidths=.5,
                            cbar=False)

        g.figure.axes[p].set_xticklabels(features_pretty, rotation='vertical', fontsize=13)
        g.figure.axes[p].set_yticklabels(features_pretty, rotation='horizontal', fontsize=13)

        g.figure.axes[p].set_xlabel('Removed feature')
        if p == 0:
            g.figure.axes[p].set_ylabel('Remaining features')

        g.figure.axes[p].set_title(predictors_pretty[p])

        ### with pyplot:
        #masked_array = np.ma.masked_where(np.transpose(importance) == value, np.transpose(importance))
        #plt.imshow(masked_array, cmap=cmap)

    plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_importance.png')
    plt.close()
