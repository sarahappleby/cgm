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
    line = sys.argv[4]

    predictors = ['rho', 'T', 'Z']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    limit_dict = {}
    limit_dict['rho'] = [[0, 4], [2, 4], [2, 4], [1.8, 3.8], [1, 3.5], [0.5, 3.25]]
    limit_dict['T'] = [[3, 6], [3.5, 4.75], [4, 5], [4, 5], [4, 5.3], [4.25, 6]]
    limit_dict['Z'] = [[-4, 1], [-0.75, 0.75], [-0.75, 0.75], [-0.75, 0.75], [-0.75, 0.75], [-0.75, 0.75]]
    nbins = 20
    gridsize = 25
    
    xlabels = [r'${\rm log}\ \delta_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    ylabels = [r'${\rm log}\ \delta_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']
    x_dict = {}
    x_dict['rho'] = [0.18, 0.2, 0.2, 0.18, 0.18, 0.18]
    x_dict['T'] = [0.18, 0.18, 0.18, 0.18, 0.18, 0.2]
    x_dict['Z'] = [0.18, 0.23, 0.23, 0.23, 0.23, 0.23]

    snapfile = f'/disk04/sapple/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'
    #model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models_app2022_data/'

    diff = {pred: None for pred in predictors}
    err = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

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

        diff[pred] = np.array(data[pred]) - np.array(data[f'{pred}_pred'])

        coords = np.transpose(np.array([data[pred], data[f'{pred}_pred']]))
        d_perp = np.cross(points[1] - points[0], points[0] - coords) / np.linalg.norm(points[1]-points[0])

        scores = {}
        scores['Predictor'] = pred
        scores['Pearson'] = round(pearsonr(data[pred],data[pred+'_pred'])[0], 5)
        scores['sigma_perp'] = round(np.nanstd(d_perp), 5)
        for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
            scores[_scorer.__name__] = float(_scorer(data[pred],
                                               data[pred_str], multioutput='raw_values'))
        err = err.append(scores, ignore_index=True) 

        dx = (limits[1] - limits[0]) / nbins
        bins = np.arange(limits[0], limits[1]+dx, dx)

        mask = (data[pred] > bins[0]) & (data[pred] < bins[-1])
        diff_within = np.sum(diff[pred] <= 0.2) / len(diff[pred])

        g = sns.jointplot(data=data[mask], x=pred, y=f'{pred}_pred', kind="hex", 
                          joint_kws=dict(bins='log', alpha=0.8, extent=(limits[0], limits[1], limits[0], limits[1]), gridsize=gridsize), 
                          xlim=[limits[0], limits[1]], ylim=[limits[0], limits[1]],
                          marginal_ticks=True, marginal_kws=dict(bins=bins, fill=False, stat='probability'))

        g.figure.axes[0].plot(bins, bins, ls=':', lw=2, c='k')
        g.set_axis_labels(xlabel=xlabels[p], ylabel=ylabels[p])

        g.figure.axes[1].set_yticks([0.1])
        g.figure.axes[2].set_xticks([0.1])

        #annotation = r'$\sigma_\perp = $'+f'{scores["sigma_perp"]}\n'+r'$ \rho_r = $'+' {}\nPredictions within\n 0.2 dex: {}\%'.format(scores['Pearson'], diff_within)
        annotation = r'$\sigma_\perp = $'\
                     f' {scores["sigma_perp"]:.2f}\n'\
                     r'$\rho_r = $'\
                     f' {scores["Pearson"]:.2f}\n'\
                     r'${\rm MSE} = $'\
                     f' {scores["mean_squared_error"]:.2f}'\
                     #r'$f_{\leq 0.2 \rm dex} = $'\
                     #f' {diff_within:.2f}'\

        g.figure.axes[0].text(0.66, 0.05, annotation, transform=g.figure.axes[0].transAxes)

        cax = g.figure.add_axes([x_dict[pred][lines.index(line)], .6, .02, .2])
        cbar = g.figure.colorbar(mpl.cm.ScalarMappable(norm=g.figure.axes[0].collections[0].norm, cmap=g.figure.axes[0].collections[0].cmap),
                                 cax=cax)
        cbar.set_label(r'$n$', rotation='horizontal')

        plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_joint_single_{pred}.png')
        plt.close()
    
    print(err)
