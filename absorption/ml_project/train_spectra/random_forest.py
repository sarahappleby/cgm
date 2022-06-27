### Routine to apply the scipy random forest to the training data

import h5py
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

from scipy.stats import pearsonr

if __name__ == '__main__':


    # Step 1) read in the training data
    # Step 2) treat the data such that unphysical/awkward values are taken care of
    # Step 3) Scale the data such that means are zero and variance is 1
    preprocessing.StandardScaler().fit
    # Step 4) Cross validation of the random forest using Kfold
    # Step 5) Run the random forest routine

    split = 0.6
    train = np.random.rand(len(sample)) < split
