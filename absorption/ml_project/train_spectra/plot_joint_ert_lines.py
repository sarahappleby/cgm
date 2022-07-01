import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = sys.argv[4]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    limits = [[-30, -26], [3, 7], [-4, 0]]
    xlabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    ylabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']
    x = [0.22, 0.17, .2]

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    etree, features, predictors, feature_scaler, predictor_scaler, df_full = \
                pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_ERT.model', 'rb'))
    train = df_full['train_mask']

    test_data = df_full[~train]; del df_full
    test_data = test_data.reset_index(drop=True)
    prediction = pd.DataFrame(predictor_scaler.inverse_transform(etree.predict(feature_scaler.transform(test_data[features]))),
                              columns=[pred+'_pred' for pred in predictors])
    data = pd.concat([test_data[predictors], prediction], axis=1); del prediction

    for p, pred in enumerate(predictors):

        bins = np.arange(limits[p][0], limits[p][1]+0.2, 0.2)

        mask = (data[pred] > bins[0]) & (data[pred] < bins[-1])

        g = sns.jointplot(data=data[mask], x=pred, y=f'{pred}_pred', 
                          kind="hex", joint_kws=dict(bins='log', alpha=0.8), xlim=[limits[p][0], limits[p][1]], ylim=[limits[p][0], limits[p][1]],
                          marginal_ticks=True, marginal_kws=dict(bins=bins, fill=False, stat='probability'))
        """
        g = sns.JointGrid(data=data[mask], x=pred, y=f'{pred}_pred', marginal_ticks=True)
        g.plot_marginals(sns.histplot, bins=bins, fill=False, stat='probability')
        g.plot_joint(plt.hexbin)
        """

        g.figure.axes[0].plot(bins, bins, ls=':', lw=2, c='k')
        g.set_axis_labels(xlabel=xlabels[p], ylabel=ylabels[p])

        g.figure.axes[1].set_yticks([0.1])
        g.figure.axes[2].set_xticks([0.1])

        cax = g.figure.add_axes([x[p], .6, .02, .2])
        g.figure.colorbar(mpl.cm.ScalarMappable(norm=g.figure.axes[0].collections[0].norm, cmap=g.figure.axes[0].collections[0].cmap),
                          cax=cax, label=r'$n$')

        """
        pos = g.figure.axes[1].get_position()
        pos = matplotlib.transforms.Bbox([[pos.xmin, pos.ymin+0.025], [pos.xmax, pos.ymax+0.025]])
        g.figure.axes[1].set_position(pos)

        pos = g.figure.axes[2].get_position()
        pos = matplotlib.transforms.Bbox([[pos.xmin+0.04, pos.ymin], [pos.xmax+0.04, pos.ymax]])
        g.figure.axes[2].set_position(pos)
        """
        plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_ERT_joint_{pred}.png')
        plt.close()
