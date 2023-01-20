### Routine to apply the sklearn randomm forest to the line by line absorption data

import h5py
import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

from scipy.stats import pearsonr

np.random.seed(1)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    line = sys.argv[4]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    chisq_lim = [4., 50., 15.8, 39.8, 8.9, 4.5]
    N_min = [12.7, 11.5, 12.8, 11.7, 12.8, 13.2]

    delta_fr200 = 0.25
    min_fr200 = 0.25
    nbins_fr200 = 5
    fr200 = np.arange(min_fr200, (nbins_fr200+1)*delta_fr200, delta_fr200)

    features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot']
    predictors = ['rho', 'T', 'Z']

    # Step 1) read in the training data
    sample_dir = f'/disk04/sapple/data/samples/'
    with h5py.File(f'{sample_dir}{model}_{wind}_{snap}_galaxy_sample.h5', 'r') as sf:
        gal_ids = sf['gal_ids'][:]
        mass = sf['mass'][:]
        ssfr = sf['ssfr'][:]
        kappa_rot = sf['kappa_rot'][:]
    
    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'
    results_file = f'/disk04/sapple/data/normal/results/{model}_{wind}_{snap}_fit_lines_{line}.h5'
    
    dataset = {}
    dataset['rho'] = []
    dataset['T'] = []
    dataset['Z'] = []
    dataset['N'] = []
    dataset['b'] = []
    dataset['EW'] = []
    dataset['chisq'] = []
    dataset['dv'] = []
    dataset['gal_ids'] = []
    dataset['r_perp'] = []
    
    for i in range(len(fr200)):
        with h5py.File(results_file, 'r') as hf:
            dataset['rho'].extend(hf[f'log_rho_{fr200[i]}r200'][:])
            dataset['T'].extend(hf[f'log_T_{fr200[i]}r200'][:])
            dataset['Z'].extend(hf[f'log_Z_{fr200[i]}r200'][:])
            dataset['N'].extend(hf[f'log_N_{fr200[i]}r200'][:])
            dataset['b'].extend(hf[f'b_{fr200[i]}r200'][:])
            dataset['EW'].extend(hf[f'ew_{fr200[i]}r200'][:])
            dataset['chisq'].extend(hf[f'chisq_{fr200[i]}r200'][:])
            dataset['dv'].extend(hf[f'pos_dv_{fr200[i]}r200'][:])
            dataset['gal_ids'].extend(hf[f'ids_{fr200[i]}r200'][:])
            dataset['r_perp'].extend([fr200[i]] * len(hf[f'ids_{fr200[i]}r200'][:]))

    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])

    mask = (dataset['N'] > N_min[lines.index(line)]) * (dataset['chisq'] < chisq_lim[lines.index(line)]) * (dataset['b'] > 0)
    for key in dataset.keys():
        dataset[key] = dataset[key][mask]

    idx = np.array([np.where(gal_ids == l)[0] for l in dataset['gal_ids']]).flatten() 
    dataset['mass'] = mass[idx]
    dataset['ssfr'] = ssfr[idx]
    dataset['kappa_rot'] = kappa_rot[idx]

    # Step 2) treat the data such that unphysical/awkward values are taken care of
    dataset['EW'] = np.log10(dataset['EW'] + 1e-3)
    dataset['b'] = np.log10(dataset['b'] + 1)
    df_full = pd.DataFrame(dataset); del dataset
     
    # Step 3) Scale the data such that means are zero and variance is 1
    split = 0.8
    train = np.random.rand(len(df_full)) < split
    df_full['train_mask'] = train

    print("train / test:", np.sum(train), np.sum(~train))

    feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features])
    predictor_scaler = preprocessing.StandardScaler().fit(df_full[train][predictors])

    # Step 4) Cross validation of the random forest using Kfold
    ss = KFold(n_splits=5, shuffle=True)
    tuned_parameters = {'n_neighbors':[2, 5, 10, 15, 20],
                        'weights':['uniform', 'distance'],
                        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size':[20, 30, 40, 50]
                        }

    knn = GridSearchCV(KNeighborsRegressor(), param_grid=tuned_parameters, refit=True, cv=None, n_jobs=4)

    # Step 5) Run and save the KNN routine
    knn.fit(feature_scaler.transform(df_full[train][features]), predictor_scaler.transform(df_full[train][predictors]))
    pickle.dump([knn, features, predictors, feature_scaler, predictor_scaler, df_full], 
                open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_KNN.model', 'wb'))

    # Step 6) Predict conditions
    conditions_pred = pd.DataFrame(predictor_scaler.inverse_transform(knn.predict(feature_scaler.transform(df_full[~train][features]))),columns=predictors)
    conditions_true = pd.DataFrame(df_full[~train],columns=predictors)

    # Step 7) Evaluate performance
    pearson = []
    for p in predictors:
        pearson.append(round(pearsonr(df_full[~train][p],conditions_pred[p])[0],3))
    err = pd.DataFrame({'Predictors': conditions_pred.columns, 'Pearson': pearson})

    scores = {}
    for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
        err[_scorer.__name__] = _scorer(df_full[~train][predictors],
                                        conditions_pred, multioutput='raw_values')
    print(err)
