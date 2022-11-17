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
rng = np.random.RandomState(0)

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

    Xtrain = np.zeros((len(df_full[train]), len(features)))
    Xtest = np.zeros((len(df_full[~train]), len(features)))
    for i in range(len(features)): 
        Xtrain[:, i] = df_full[train][features[i]]
        Xtest[:, i] = df_full[~train][features[i]]

    qt = preprocessing.QuantileTransformer(output_distribution="normal", random_state=rng)
    qt_fit = qt.fit(Xtrain)
    qt_transformed = qt_fit.transform(Xtest)

    # to transform back:
    qt.inverse_tranform(qt_transformed)

    # Step 2) Scale the data such that means are zero and variance is 1
    feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features])
    predictor_scaler = preprocessing.StandardScaler().fit(np.array(df_full[train][predictor]).reshape(-1, 1) )

