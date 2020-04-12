#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import time

import matplotlib.pyplot as plt
import numpy as np


def register_act(experiment, dp_logs=True, shuffle=False):
    """ Gets network activations when presented with inputs for each class and runs
     within- and between-batch pearson correlation and dot products between classes.
    :param dp_logs (optional): Boolean, determines whether to return
    the dot product log (base 10)
    :param shuffle (optional): Boolean, if True will also return the metrics
     calculated on a shuffled version of the activations
    """

    layer_list = list(experiment.model.named_children())
    layer_names = [p[0] for p in layer_list]

    def get_act(name):
        def hook(model, input_, output):
            act[name] = output.detach().cpu().numpy()
        return hook

    cnt = 0
    for module in experiment.model:
        module.register_forward_hook(get_act(layer_names[cnt]))
        cnt += 1

    outputs = []
    for k in range(1, 11):
        act = {}
        loader = experiment.test_loader[k]
        x, _ = next(iter(loader))
        experiment.model(x.cuda())
        outputs.append(act)

    all_keys = outputs[0].keys()

    corr_mats, dot_mats = [], []
    shuffled_corr_mats, shuffled_dot_mats = [], []
    for key in all_keys:
        mod_output = [np.vstack([outputs[n][key][k, :].flatten()
                                 for k in range(experiment.batch_size)])
                      for n in range(len(outputs))]

        m_len = mod_output[0].shape[1]
        iu = np.triu_indices(experiment.batch_size, 1)
        off_indices = np.triu_indices(10, 1)

        corr_mat = np.zeros((10, 10))
        dot_mat = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                corr_mat[i, j] = np.corrcoef(mod_output[i], mod_output[j])[iu].mean()
                dot_mat[i, j] = np.nanmean([np.dot(mod_output[i][x, :],
                                                   mod_output[j][y, :]) / m_len
                                            for x in range(experiment.batch_size)
                                            for y in range(experiment.batch_size)])

        corr_mats.append(corr_mat)
        dot_mats.append(dot_mat)
        offdiag_corrs = [np.nanmean(cc[off_indices]) for cc in corr_mats]
        diag_corrs = [np.nanmean(np.diag(cc)) for cc in corr_mats]

        if shuffle:
            shuff_corr_mat = np.zeros((10, 10))
            shuff_dot_mat = np.zeros((10, 10))
            shuffled_outputs = [np.random.permutation(k.T).T
                                for k in mod_output]

            for i in range(10):
                for j in range(10):
                    shuff_corr_mat[i, j] = np.corrcoef(
                        shuffled_outputs[i], shuffled_outputs[j])[iu].mean()
                    shuff_dot_mat[i, j] = np.nanmean(
                        [np.dot(shuffled_outputs[i][x, :],
                                shuffled_outputs[j][y, :]) / m_len
                         for x in range(experiment.batch_size)
                         for y in range(experiment.batch_size)])

            shuffled_corr_mats.append(shuff_corr_mat)
            shuffled_dot_mats.append(shuff_dot_mat)

            sh_offdiag_corrs = [np.nanmean(cc_[off_indices]) for cc_ in shuffled_corr_mats]
            sh_diag_corrs = [np.nanmean(np.diag(cc_)) for cc_ in shuffled_corr_mats]

    def try_log(x):
        out = np.log10(x)
        if np.isnan(out) or np.isinf(out):
            pass
        else:
            return out

    if dp_logs:
        offdiag_dotprods = [try_log(np.nanmean(dp[off_indices]) + 1e-9)
                            for dp in dot_mats]
        diag_dotprods = [try_log(np.nanmean(np.diag(dp)) + 1e-9) for dp in dot_mats]
        if shuffle:
            sh_offdiag_dotprods = [try_log(np.nanmean(dp_[off_indices]) + 1e-9)
                                   for dp_ in shuffled_dot_mats]
            sh_diag_dotprods = [try_log(np.nanmean(np.diag(dp_)) + 1e-9)
                                for dp_ in shuffled_dot_mats]

    else:
        offdiag_dotprods = [np.nanmean(dp[off_indices]) for dp in dot_mats]
        diag_dotprods = [np.nanmean(np.diag(dp)) for dp in dot_mats]

        if shuffle:
            sh_offdiag_dotprods = [np.nanmean(dp_[off_indices])
                                   for dp_ in shuffled_dot_mats]
            sh_diag_dotprods = [np.nanmean(np.diag(dp_)) for dp_ in shuffled_dot_mats]

    corrs_ = [offdiag_corrs, diag_corrs, offdiag_dotprods, diag_dotprods]

    if shuffle:
        shuffled_corrs = [sh_offdiag_corrs, sh_diag_corrs,
                          sh_offdiag_dotprods, sh_diag_dotprods]
        return corrs_, shuffled_corrs
    else:
        return corrs_


def plot_metrics(metrics, order=[0, 2, 4, 6, 9, 11], savefig=False,
                 savestring=None,
                 legend_=None):
    metrics_list = []
    for metric in metrics:
        metric_list = [[N[k] for k in order] for N in metric]
        metrics_list.append(np.array(metric_list).T)

    metrics_list = np.array(metrics_list)
    metric_divider = len(metrics) / 2 - 1
    ylim_ = [np.nanmin([np.nanmin(x) for x in metrics_list]) - 0.5,
             np.nanmax([np.nanmax(x) for x in metrics_list]) + 0.5]

    ylim_[ylim_ == -np.inf] = np.sort([np.nanmin(x) for x in metrics_list])[0] - 0.5

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    cnt = 0
    for axis in np.hstack(ax):
        array_ = metrics_list[cnt]
        axis.plot(array_, "o", alpha=0.6)

        module_keys = ["cnn1", "cnn1_actfn", "cnn2",
                       "cnn2_actfn", "linear1", "linear1_actfn"]
        metric_names = ["Pearson off-diag", "Pearson diag",
                        "Dot product off-diag", "Dot product diag"]
        ylabels = ["Pearson correlation", "log norm. dot product"]

        axis.set_xticks(range(6))
        axis.set_title(metric_names[cnt])
        axis.set_ylabel(ylabels[int(cnt / 2)], fontsize=11)

        if cnt == 0:
            if legend_ is not None:
                axis.legend(legend_, frameon=False)
            else:
                if array_.shape[1] == 2:
                    axis.legend(["Dense CNN", "Sparse CNN"], frameon=False)
                else:
                    axis.legend([int(64 / a) for a in [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]],
                                frameon=False)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if cnt > metric_divider:
            axis.axhline([0], color="k", linestyle="--")
            axis.set_ylim(ylim_)
            axis.set_xticklabels(module_keys, rotation=60)

        else:
            axis.set_ylim((0., 1.))
            axis.set_xticklabels("")

        cnt += 1

    if savefig:
        if savestring is not None:
            plt.savefig("../plots/{}.pdf".format(savestring))
        else:
            print("adding timestamp")
            timestamp = time.time()
            plt.savefig("../plots/corr_quantifications_{}.pdf".format(timestamp))
