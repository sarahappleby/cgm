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

    features = ['EW_HI', 'EW_MgII', 'EW_CII', 'EW_SiIII', 'EW_CIV', 'EW_OVI', 'fr200', 'mass', 'ssfr', 'kappa_rot']
    features_pretty = [r'${\rm EW}_{\rm HI}$', r'${\rm EW}_{\rm MgII}$', r'${\rm EW}_{\rm CII}$',
                       r'${\rm EW}_{\rm SiIII}$', r'${\rm EW}_{\rm CIV}$', r'${\rm EW}_{\rm OVI}$',
                       r'$f_{r200}$', r'$M_\star$',
                       r'${\rm sSFR}$', r'$\kappa_{\rm rot}$']

    features = ['EW_HI', 'EW_MgII', 'EW_CII', 'EW_SiIII', 'EW_CIV', 'EW_OVI', 'fr200']
    features_pretty = [r'${\rm EW}_{\rm HI}$', r'${\rm EW}_{\rm MgII}$', r'${\rm EW}_{\rm CII}$',
                       r'${\rm EW}_{\rm SiIII}$', r'${\rm EW}_{\rm CIV}$', r'${\rm EW}_{\rm OVI}$',
                       r'$f_{r200}$']

    predictors = ['log_fgas_cool', 'log_fgas_warm', 'log_fgas_hot']
    predictors_pretty = [r'${\rm log}\ f_{\rm cool}$', r'${\rm log}\ f_{\rm warm}$', r'${\rm log}\ f_{\rm hot}$']

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    cmap = sns.color_palette("flare_r", as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 9))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.295, 0.02, 0.4])

    importance = np.zeros((3, len(features), len(features)))

    for p, pred in enumerate(predictors):

        gridsearch, _, _, _, _, df_full = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_ew_RF_{pred}.model', 'rb'))
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
            g = sns.heatmap(importance_use, mask=mask, cmap=cmap, vmax=1, vmin=0, annot=True, annot_kws={'fontsize':12.5}, ax=ax[p], 
                            square=True, linewidths=.5, cbar_ax=cbar_ax)
        else:
            g = sns.heatmap(importance_use, mask=mask, cmap=cmap, vmax=1, vmin=0, annot=True, annot_kws={'fontsize':12.5}, ax=ax[p], 
                            square=True, linewidths=.5, cbar=False)

        g.figure.axes[p].set_xticklabels(features_pretty, rotation='vertical', fontsize=13)
        g.figure.axes[p].set_yticklabels(features_pretty, rotation='horizontal', fontsize=13)

        g.figure.axes[p].set_xlabel('Removed feature')
        if p == 0:
            g.figure.axes[p].set_ylabel('Feature importance')

        g.figure.axes[p].set_title(predictors_pretty[p])

        ### with pyplot:
        #masked_array = np.ma.masked_where(np.transpose(importance) == value, np.transpose(importance))
        #plt.imshow(masked_array, cmap=cmap)

    plt.savefig(f'plots/{model}_{wind}_{snap}_ew_RF_importance.png')
    plt.close()
