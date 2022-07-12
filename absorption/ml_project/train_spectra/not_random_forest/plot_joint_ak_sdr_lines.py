import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
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
    line = sys.argv[4]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    limits = [[-30, -26], [3, 6.4], [-4, -1]]
    xlabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    ylabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']
    x = [0.22, 0.17, .2]

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    knn, features, predictors, feature_scaler, predictor_scaler, df_full = \
                pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_AK_SDR.model', 'rb'))
    train = df_full['train_mask']
    pred_str = [p+'_pred' for p in predictors]

    test_data = df_full[~train]; del df_full
    test_data = test_data.reset_index(drop=True)
    prediction = pd.DataFrame(predictor_scaler.inverse_transform(knn.predict(feature_scaler.transform(test_data[features]))),
                              columns=[pred+'_pred' for pred in predictors])
    data = pd.concat([test_data[predictors], prediction], axis=1); del prediction

    diff = {pred: None for pred in predictors}
    for pred in predictors:
        diff[pred] = np.array(data[pred]) - np.array(data[f'{pred}_pred'])

    for p, pred in enumerate(predictors):

        bins = np.arange(limits[p][0], limits[p][1]+0.2, 0.2)

        mask = (data[pred] > bins[0]) & (data[pred] < bins[-1])
        diff_within = round(100* np.sum(diff[pred] < 0.2) / len(diff[pred]) )

        g = sns.jointplot(data=data[mask], x=pred, y=f'{pred}_pred', 
                          kind="hex", joint_kws=dict(bins='log', alpha=0.8), xlim=[limits[p][0], limits[p][1]], ylim=[limits[p][0], limits[p][1]],
                          marginal_ticks=True, marginal_kws=dict(bins=bins, fill=False, stat='probability'))

        g.figure.axes[0].plot(bins, bins, ls=':', lw=2, c='k')
        g.set_axis_labels(xlabel=xlabels[p], ylabel=ylabels[p])

        g.figure.axes[1].set_yticks([0.1])
        g.figure.axes[2].set_xticks([0.1])

        g.figure.axes[0].text(0.56, 0.05, 'Predictions within\n 0.2 dex: {}\%'.format(diff_within),
                              transform=g.figure.axes[0].transAxes)

        cax = g.figure.add_axes([x[p], .6, .02, .2])
        cbar = g.figure.colorbar(mpl.cm.ScalarMappable(norm=g.figure.axes[0].collections[0].norm, cmap=g.figure.axes[0].collections[0].cmap),
                                 cax=cax, label=r'$n$')
        plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_AK_SDR_joint_{pred}.png')
        plt.close()

    # Evaluate performance
    pearson = []
    for p in predictors:
        pearson.append(round(pearsonr(data[p],data[f'{p}_pred'])[0],3))
    err_ak = pd.DataFrame({'Predictors': predictors, 'Pearson': pearson})

    scores = {}
    for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
        err_ak[_scorer.__name__] = _scorer(data[predictors],
                                            data[pred_str], multioutput='raw_values')
    print(err_ak)

