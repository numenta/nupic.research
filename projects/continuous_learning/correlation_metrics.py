import torch
import numpy as np 

from cont_speech_experiment import ContinuousSpeechExperiment, ClasswiseDataset 

import matplotlib.pyplot as plt 


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
        mod_output = [np.vstack([outputs[n][key][k,:].flatten()
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
    offdiag_corrs = [np.mean(cc[off_indices]) for cc in corr_mats]
    diag_corrs = [np.mean(np.diag(cc)) for cc in corr_mats]

    if dp_logs:
        offdiag_dotprods = [np.log(np.mean(dp[off_indices])) for dp in dot_mats]
        diag_dotprods = [np.log(np.mean(np.diag(dp))) for dp in dot_mats]
    else:
        offdiag_dotprods = [np.mean(dp[off_indices]) for dp in dot_mats]
        diag_dotprods = [np.mean(np.diag(dp)) for dp in dot_mats]

    return offdiag_corrs, diag_corrs, offdiag_dotprods, diag_dotprods


def plot_metrics(metrics, order=[0, 2, 4, 6, 9, 11]):
    metrics_list = []
    for metric in metrics:
        metric_list = [[N[k] for k in order] for N in metric]
        metrics_list.append(np.array(metric_list).T)

    metrics_list = np.array(metrics_list)
    metric_divider = len(metrics) / 2

    fig, ax = plt.subplots(2,2, figsize=(10,10))
    cnt = 0
    for axis in np.hstack(ax):
        array_ = metrics_list[cnt]
        axis.plot(array_, "o", alpha=0.6)

        module_keys = ["cnn1", "cnn1_actfn", "cnn2", "cnn2_actfn", "linear1", "linear1_actfn"]
        metric_names = ["Pearson off-diag", "Pearson diag", "Dot product off-diag", "Dot product diag"]
        ylabels = ["Pearson correlation", "log norm. dot product"]
        plt.xticks(range(6), module_keys, rotation=80)
        # plt.ylim((-8.3, 5))
        plt.title(metric_names[cnt])
        plt.ylabel(ylabels[int(cnt / 2)], fontsize=11)
        plt.axhline([0], color="k", linestyle="--")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


        cnt += 1
