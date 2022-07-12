### Routine to apply the sklearn randomm forest to the line by line absorption data

import h5py
import numpy as np
import pandas as pd
import pickle
import sys

from tpot import TPOTRegressor

from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

from scipy.stats import pearsonr

np.random.seed(1)

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    
    line = sys.argv[4]

    generations = 100
    population_size=100
    cv = 5
    random_state = 1
    verbosity = 2

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']

    features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot']
    predictor = 'Z'

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'
    export_script = f'tpot/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_tpot_{predictor}.py'

    # Step 1) read in the training data
    df_full = pd.read_csv(f'data/{model}_{wind}_{snap}_{line}_lines.csv')
    train = df_full['train_mask']

    # Step 2) Scale the data such that means are zero and variance is 1
    #feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features])
    #predictor_scaler = preprocessing.StandardScaler().fit(np.array(df_full[train][predictor]).reshape(-1, 1) )

    # Step 3) Set up and run the TPOT optimizer to find the best tree-based pipeline
    pipeline_optimizer = TPOTRegressor(generations=generations,
                                       population_size=population_size,
                                       cv=cv,
                                       random_state=random_state,
                                       verbosity=verbosity)
    pipeline_optimizer.fit(df_full[train][features], df_full[train][predictor])
    #pipeline_optimizer.fit(feature_scaler.transform(df_full[train][features]), predictor_scaler.transform(np.array(df_full[train][predictor]).reshape(-1, 1) ))
    print(pipeline_optimizer.score(df_full[~train][features], df_full[~train][predictor]))
    pipeline_optimizer.export(export_script)

    # Step 4) Predict conditions
    conditions_pred = pipeline_optimizer.predict(df_full[~train][features] )
    #conditions_pred = predictor_scaler.inverse_transform(np.array( pipeline_optimizer.predict(feature_scaler.transform(df_full[~train][features]))).reshape(-1, 1) )
    conditions_pred = pd.DataFrame(conditions_pred,columns=[predictor])
    conditions_true = pd.DataFrame(df_full[~train],columns=[predictor])

    # Step 5) Evaluate performance
    pearson = round(pearsonr(df_full[~train][predictor],conditions_pred[predictor])[0],3)
    err = pd.DataFrame({'Predictors': conditions_pred.columns, 'Pearson': pearson})

    scores = {}
    for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
        err[_scorer.__name__] = _scorer(df_full[~train][predictor],
                                        conditions_pred, multioutput='raw_values')
    print(err)
