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

import matplotlib.pyplot as plt
import numpy as np
import time


def register_act(experiment, dp_logs=True):
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

    corr_mats = []
    dot_mats = []
    for key in all_keys:
        mod_output = [np.vstack([outputs[n][key][k, :].flatten()
                                for k in range(experiment.batch_size)])
                      for n in range(len(outputs))]

        corr_mat = np.zeros((10, 10))
        iu = np.triu_indices(experiment.batch_size, 1)
        dot_mat = np.zeros((10, 10))

        m_len = mod_output[0].shape[1]

        for i in range(10):
            for j in range(10):
                corr_mat[i, j] = np.corrcoef(mod_output[i], mod_output[j])[iu].mean()
                dot_mat[i, j] = np.mean([np.dot(mod_output[i][x, :],
                                        mod_output[j][y, :]) / m_len
                                        for x in range(experiment.batch_size)
                                        for y in range(experiment.batch_size)])

        corr_mats.append(corr_mat)
        dot_mats.append(dot_mat)

    off_indices = np.triu_indices(10, 1)
    print(off_indices)
    offdiag_corrs = [np.mean(cc[off_indices]) for cc in corr_mats]
    diag_corrs = [np.mean(np.diag(cc)) for cc in corr_mats]

    if dp_logs:
        offdiag_dotprods = [np.log(np.mean(dp[off_indices])) for dp in dot_mats]
        diag_dotprods = [np.log(np.mean(np.diag(dp))) for dp in dot_mats]
    else:
        offdiag_dotprods = [np.mean(dp[off_indices]) for dp in dot_mats]
        diag_dotprods = [np.mean(np.diag(dp)) for dp in dot_mats]

    return offdiag_corrs, diag_corrs, offdiag_dotprods, diag_dotprods


def plot_metrics(metrics, order=[0, 2, 4, 6, 9, 11], savefig=False):
    metrics_list = []
    for metric in metrics:
        metric_list = [[N[k] for k in order] for N in metric]
        metrics_list.append(np.array(metric_list).T)

    metrics_list = np.array(metrics_list)
    metric_divider = len(metrics) / 2 - 1
    ylim_ = [np.min([np.min(x) for x in metrics_list]) - 0.5,
             np.max([np.max(x) for x in metrics_list]) + 0.5]

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
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if cnt > metric_divider:
            axis.axhline([0], color="k", linestyle="--")
            axis.set_ylim(ylim_)
            axis.set_xticklabels(module_keys, rotation=60)

        else:
            axis.set_ylim((0., 1.))

        cnt += 1

    if savefig:
        timestamp = time.time()
        plt.savefig("../plots/corr_quantifications_{}".format(timestamp))