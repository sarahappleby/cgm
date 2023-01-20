import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pygad as pg
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

    predictors = ['rho', 'T', 'Z']

    cmap = plt.get_cmap('Blues')
    cmap = truncate_colormap(cmap, 0., 0.9)

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    limit_dict = {}
    limit_dict['rho'] = [[0, 4], [2, 4], [2, 4], [1.5, 4], [1, 3.5], [0.5, 3.5]]
    limit_dict['T'] = [[3, 6.5], [3.5, 5], [4, 5], [4, 5], [4, 5.5], [4, 6]]
    limit_dict['Z'] = [[-4, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

    xlabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    ylabels = [r'${\rm log}\ (\rho/{\rm cm}^{-2})_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']

    snapfile = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    fig, ax = plt.subplots(2, 8, figsize=(17, 7),
                               gridspec_kw={'height_ratios': [1, 4], 'width_ratios':[2, 1, 0.5, 2, 1, 0.5, 2, 1]})

    ax[1][0].sharex(ax[0][0]); ax[1][3].sharex(ax[0][3]); ax[1][6].sharex(ax[0][6])
    ax[1][0].sharey(ax[1][1]); ax[1][3].sharey(ax[1][4]); ax[1][6].sharey(ax[1][7])

    i = 0

    for p, pred in enumerate(predictors):

        limits = limit_dict[pred][lines.index(line)]
        points = np.repeat(limits, 2).reshape(2, 2)

        random_forest, features, _, feature_scaler, predictor_scaler, df_full = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))
        train = df_full['train_mask']
        pred_str = pred+'_pred'

        test_data = df_full[~train]; del df_full
        test_data = test_data.reset_index(drop=True)
        prediction = pd.DataFrame(predictor_scaler.inverse_transform( np.array(random_forest.predict(feature_scaler.transform(test_data[features])).reshape(-1, 1) )),
                                  columns=[pred+'_pred'])
        data = pd.concat([test_data[pred], prediction], axis=1); del prediction

        if pred == 'rho':
            data[pred] -= np.log10(cosmic_rho)
            data[f'{pred}_pred'] -= np.log10(cosmic_rho)

        if pred == 'Z':
            data[pred] -= np.log10(zsolar[lines.index(line)])
            data[f'{pred}_pred'] -= np.log10(zsolar[lines.index(line)])
        
        ax[1][i].hexbin(data[pred], data[f'{pred}_pred'], cmap=cmap, bins='log')
        ax[1][i].plot(np.arange(limits[0], limits[1]+1), np.arange(limits[0], limits[1]+1), c='k', ls=':', lw=1)
        
        ax[1][i].set_xlim(limits)
        ax[1][i].set_ylim(limits)
        ax[1][i].set_xlabel(xlabels[p])
        ax[1][i].set_ylabel(ylabels[p])

        ax[0][i].hist(data[pred], bins=20, stacked=True, density=True, color='steelblue', histtype='step', ls='-', lw=1)
        ax[0][i].spines.right.set_visible(False)
        ax[0][i].spines.top.set_visible(False)
        
        ax[1][i+1].hist(data[f'{pred}_pred'], bins=20, stacked=True, density=True, color='steelblue', histtype='step', ls='-', lw=1, orientation='horizontal')
        ax[1][i+1].spines.right.set_visible(False)
        ax[1][i+1].spines.top.set_visible(False)        

        i +=3

    fig.subplots_adjust(wspace=0.1, hspace=0.1) 

    fig.delaxes(ax[0][1]); fig.delaxes(ax[0][2]); fig.delaxes(ax[0][4]); fig.delaxes(ax[0][5]); fig.delaxes(ax[0][7])
    fig.delaxes(ax[1][2]); fig.delaxes(ax[1][5])

    plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_joint.png')
