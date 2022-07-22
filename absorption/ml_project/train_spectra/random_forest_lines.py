### Routine to apply the sklearn randomm forest to the line by line absorption data

import h5py
import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
    predictor = sys.argv[5]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot']
    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'
    n_jobs = 10
    
    # Step 1) read in the training data
    df_full = pd.read_csv(f'data/{model}_{wind}_{snap}_{line}_lines.csv')
    train = df_full['train_mask']

    # Step 2) Scale the data such that means are zero and variance is 1
    feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features])
    predictor_scaler = preprocessing.StandardScaler().fit(np.array(df_full[train][predictor]).reshape(-1, 1) )

    # Step 3) Cross validation of the random forest using Kfold
    ss = KFold(n_splits=5, shuffle=True)
    tuned_parameters = {'n_estimators':np.arange(10, 155, 5),
                        'min_samples_split': np.arange(5, 55, 5), 
                        'min_samples_leaf': np.arange(2, 12, 2), 
                        } 

    random_forest = GridSearchCV(RandomForestRegressor(), param_grid=tuned_parameters, refit=True, cv=None, n_jobs=n_jobs)

    # Step 4) Run and save the random forest routine
    random_forest.fit(feature_scaler.transform(df_full[train][features]), predictor_scaler.transform(np.array(df_full[train][predictor]).reshape(-1, 1) ))
    print(random_forest.best_params_)    
    pickle.dump([random_forest, features, predictor, feature_scaler, predictor_scaler, df_full], 
                open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{predictor}.model', 'wb'))

    # Step 5) Predict conditions
    conditions_pred = predictor_scaler.inverse_transform(np.array( random_forest.predict(feature_scaler.transform(df_full[~train][features]))).reshape(-1, 1) )
    conditions_pred = pd.DataFrame(conditions_pred,columns=[predictor])
    conditions_true = pd.DataFrame(df_full[~train],columns=[predictor])

    # Step 6) Evaluate performance
    pearson = round(pearsonr(df_full[~train][predictor],conditions_pred[predictor])[0],3)
    err = pd.DataFrame({'Predictors': conditions_pred.columns, 'Pearson': pearson})

    scores = {}
    for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
        err[_scorer.__name__] = _scorer(df_full[~train][predictor],
                                        conditions_pred, multioutput='raw_values')
    print(err)

    # Step 7) Feature importance
    importance_rf = random_forest.best_estimator_.feature_importances_
    idx = importance_rf.argsort()[::-1]
    print(np.asarray(features)[idx])
