import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1)
import matplotlib.collections as clt
import ptitprince as pt
import matplotlib.gridspec as gridspec


def hyperparameter_search_panel():
    """
    New graph after fixing error in understanding processing in analyze_result
    and re-running the 10 tasks data because it looked weird. Added 50 tasks.
    """

    savefigs = True
    figs_dir = "figs/"
    if savefigs:
        if not os.path.isdir(f"{figs_dir}"):
            os.makedirs(f"{figs_dir}")

    experiment_folder = "~/nta/nupic.research/projects/dendrites/permutedMNIST/experiments/data_hyperparameter_search/"

    df_path1 = f"{experiment_folder}segment_search.csv"
    df1 = pd.read_csv(df_path1)

    df_path2 = f"{experiment_folder}kw_sparsity_search.csv"
    df2 = pd.read_csv(df_path2)

    df_path3 = f"{experiment_folder}w_sparsity_search.csv"
    df3 = pd.read_csv(df_path3)

    df_path1_50 = f"{experiment_folder}segment_search_50.csv"
    df1_50 = pd.read_csv(df_path1_50)

    df_path2_50 = f"{experiment_folder}kw_sparsity_search_50.csv"
    df2_50 = pd.read_csv(df_path2_50)

    df_path3_50 = f"{experiment_folder}w_sparsity_search_50.csv"
    df3_50 = pd.read_csv(df_path3_50)

    df1 = df1[["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]]
    df2 = df2[["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]]
    df3 = df3[["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]]
    df1_50 = df1_50[
        ["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]
    ]
    df2_50 = df2_50[
        ["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]
    ]
    df3_50 = df3_50[
        ["Activation sparsity", "FF weight sparsity", "Num segments", "Accuracy"]
    ]

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1_50 = fig.add_subplot(gs[1, 0])
    ax2_50 = fig.add_subplot(gs[1, 1])
    ax3_50 = fig.add_subplot(gs[1, 2])

    x1 = "Num segments"
    x2 = "Activation sparsity"
    x3 = "FF weight sparsity"

    y = "Accuracy"
    ort = "v"
    pal = sns.color_palette(n_colors=6)
    sigma = 0.2
    fig.suptitle("Impact of the different hyperparameters on performance", fontsize=12)

    pt.RainCloud(
        x=x1,
        y=y,
        data=df1,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax1,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    pt.RainCloud(
        x=x1,
        y=y,
        data=df1_50,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax1_50,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    pt.RainCloud(
        x=x2,
        y=y,
        data=df2,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax2,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    pt.RainCloud(
        x=x2,
        y=y,
        data=df2_50,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax2_50,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    pt.RainCloud(
        x=x3,
        y=y,
        data=df3,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax3,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    pt.RainCloud(
        x=x3,
        y=y,
        data=df3_50,
        palette=pal,
        bw=sigma,
        width_viol=0.6,
        ax=ax3_50,
        orient=ort,
        move=0.2,
        pointplot=True,
        alpha=0.65,
    )
    ax1.set_ylabel("Mean accuracy", fontsize=14)
    ax1.set_xlabel("Number of dendritic segments", fontsize=14)
    ax1.set_ylim([0.65, 0.96])
    ax1_50.set_ylabel("Mean accuracy", fontsize=14)
    ax1_50.set_xlabel("Number of dendritic segments", fontsize=14)
    ax1_50.set_ylim([0.65, 0.96])

    ax2.set(ylabel="")
    ax2.set_xlabel("Activation density", fontsize=14)
    ax2.set_ylim([0.35, 0.96])
    # ax2.set(
    #     xticklabels=["0.99", "0.9", "0.8", "0.6", "0.4", "0.2", "0.1", "0.05", "0.01"]
    # )
    ax2_50.set(ylabel="")
    ax2_50.set_xlabel("Activation density", fontsize=14)
    ax2_50.set_ylim([0.35, 0.96])

    ax3.set(ylabel="")
    ax3.set_xlabel("FF Weight density", fontsize=14)
    ax3.set_ylim([0.4, 0.96])
    # ax3.set(xticklabels=["0.99", "0.95", "0.9", "0.5", "0.3", "0.1", "0.05", "0.01"])
    ax3_50.set(ylabel="")
    ax3_50.set_xlabel("FF Weight density", fontsize=14)
    ax3_50.set_ylim([0.4, 0.96])

    # Add 10 tasks and 50 tasks labels on the left
    plt.figtext(-0.02, 0.7, "10 TASKS", fontsize=16)
    plt.figtext(-0.02, 0.28, "50 TASKS", fontsize=16)

    fig.suptitle(
        "Impact of different hyperparameters on \n 10-tasks and 50-tasks permuted MNIST performance",
        fontsize=16,
    )
    if savefigs:
        plt.savefig(f"{figs_dir}/fourth_iteration_figure1.png", bbox_inches="tight")


if __name__ == "__main__":
    # first_iteration()
    # second_iteration()
    # third_iteration()
    # cns_figure_1c()
    fourth_iteration()
