import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pickle
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = sys.argv[4]

    cmap = plt.get_cmap('Blues')
    cmap = truncate_colormap(cmap, 0., 0.9)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    limits = [[-30, -26], [3, 7], [-4, 0]]
    xlabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    ylabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']
    bins = [np.arange]

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    random_forest, features, predictors, feature_scaler, predictor_scaler, df_full = \
                pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF.model', 'rb'))
    train = df_full['train_mask']

    conditions_pred = pd.DataFrame(predictor_scaler.inverse_transform(random_forest.predict(feature_scaler.transform(df_full[~train][features]))),columns=predictors)


    fig, ax = plt.subplots(2, 8, figsize=(17, 7), 
                           gridspec_kw={'height_ratios': [1, 2], 'width_ratios':[2, 1, 0.5, 2, 1, 0.5, 2, 1]})

    ax[1][0].sharex(ax[0][0]); ax[1][3].sharex(ax[0][3]); ax[1][6].sharex(ax[0][6])
    ax[1][0].sharey(ax[1][1]); ax[1][3].sharey(ax[1][4]); ax[1][6].sharey(ax[1][7])

    i = 0
    for p, pred in enumerate(predictors):
       
        bins = 

        ax[1][i].hexbin(df_full[pred][~train], conditions_pred[pred], cmap=cmap, bins='log')
        ax[1][i].plot(np.arange(limits[p][0], limits[p][1]+1), np.arange(limits[p][0], limits[p][1]+1), c='k', ls=':', lw=1)
        ax[1][i].set_xlim(limits[p])
        ax[1][i].set_ylim(limits[p])
        ax[1][i].set_xlabel(xlabels[p])
        ax[1][i].set_ylabel(ylabels[p])

        ax[0][i].hist(df_full[pred][~train], bins=20, stacked=True, density=True, color='steelblue', histtype='step', ls='-', lw=1)
        ax[1][i+1].hist(conditions_pred[pred], bins=20, stacked=True, density=True, color='steelblue', histtype='step', ls='-', lw=1, orientation='horizontal')
        ax[1][i+1].set_yticks([])

        i +=3

    fig.subplots_adjust(wspace=0., hspace=0.) 

    fig.delaxes(ax[0][1]); fig.delaxes(ax[0][2]); fig.delaxes(ax[0][4]); fig.delaxes(ax[0][5]); fig.delaxes(ax[0][7])
    fig.delaxes(ax[1][2]); fig.delaxes(ax[1][5])

    plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_joint.png')
