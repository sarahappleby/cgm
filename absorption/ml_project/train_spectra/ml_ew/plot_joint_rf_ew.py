import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import pygad as pg
import pickle
import sys
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    predictors = ['log_fgas_cool', 'log_fgas_warm', 'log_fgas_hot']

    limit_dict = {}
    limit_dict['log_fgas_cool'] = [-3, 0]
    limit_dict['log_fgas_warm'] = [-2, 0]
    limit_dict['log_fgas_hot'] = [-0.5, 0]
    nbins = 20

    xlabels = [r'${\rm log}\ f_{\rm cool, True}$',
               r'${\rm log}\ f_{\rm warm, True}$',
               r'${\rm log}\ f_{\rm hot, True}$',]

    ylabels = [r'${\rm log}\ f_{\rm cool, Pred}$',
               r'${\rm log}\ f_{\rm warm, Pred}$',
               r'${\rm log}\ f_{\rm hot, Pred}$',]

    x = [0.22, 0.17, .2]

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    diff = {pred: None for pred in predictors}
    err = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

    for p, pred in enumerate(predictors):

        limits = limit_dict[pred]

        random_forest, features, _, feature_scaler, predictor_scaler, df_full = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_ew_RF_{pred}.model', 'rb'))
        train = df_full['train_mask']
        pred_str = pred+'_pred'

        test_data = df_full[~train]; del df_full
        test_data = test_data.reset_index(drop=True)
        prediction = pd.DataFrame(predictor_scaler.inverse_transform( np.array(random_forest.predict(feature_scaler.transform(test_data[features])).reshape(-1, 1) )),
                                  columns=[pred+'_pred'])
        data = pd.concat([test_data[pred], prediction], axis=1); del prediction

        diff[pred] = np.array(data[pred]) - np.array(data[f'{pred}_pred'])
        
        scores = {}
        scores['Predictor'] = pred
        scores['Pearson'] = round(pearsonr(data[pred],data[pred+'_pred'])[0],3)
        for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
            scores[_scorer.__name__] = float(_scorer(data[pred],
                                               data[pred_str], multioutput='raw_values'))
        err = err.append(scores, ignore_index=True) 

        dx = (limits[1] - limits[0]) / nbins
        bins = np.arange(limits[0], limits[1]+dx, dx)

        mask = (data[pred] > bins[0]) & (data[pred] < bins[-1])
        diff_within = round(100* np.sum(diff[pred] < 0.2) / len(diff[pred]) )

        g = sns.jointplot(data=data[mask], x=pred, y=f'{pred}_pred', 
                          kind="hex", joint_kws=dict(bins='log', alpha=0.8), xlim=[limits[0], limits[1]], ylim=[limits[0], limits[1]],
                          marginal_ticks=True, marginal_kws=dict(bins=bins, fill=False, stat='probability'))

        g.figure.axes[0].plot(bins, bins, ls=':', lw=2, c='k')
        g.set_axis_labels(xlabel=xlabels[p], ylabel=ylabels[p])

        g.figure.axes[1].set_yticks([0.1])
        g.figure.axes[2].set_xticks([0.1])

        g.figure.axes[0].text(0.56, 0.05, r'$\rho_r = $'+' {}\nPredictions within\n 0.2 dex: {}\%'.format(round(scores['Pearson'], 2), diff_within),
                              transform=g.figure.axes[0].transAxes)

        cax = g.figure.add_axes([x[p], .6, .02, .2])
        g.figure.colorbar(mpl.cm.ScalarMappable(norm=g.figure.axes[0].collections[0].norm, cmap=g.figure.axes[0].collections[0].cmap),
                          cax=cax, label=r'$n$')

        plt.savefig(f'plots/{model}_{wind}_{snap}_ew_RF_joint_single_{pred}.png')
        plt.close()
    
    print(err)
