import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
sns.set(style="whitegrid",font_scale=1)
import matplotlib.collections as clt
import ptitprince as pt
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 16})


def first_iteration():
    '''
    Initial experimental runs. Missing some of the extreme values for w_sparsity or
    kw_sparsity.
    '''

    experiment_folder = '/home/jeremy/nta/results/experiments/dendrites/cns_abstract_2021/'
    df_path = f'{experiment_folder}experiment_df_summary.csv'
    df = pd.read_csv(df_path)


    savefigs = True
    figs_dir = 'figs/'
    if savefigs:
        if not os.path.isdir(f'{figs_dir}'):
            os.makedirs(f'{figs_dir}')

    # df = df[['Activation sparsity', 'FF weight sparsity', 'Accuracy']]
    df.columns
    df = df[(df.training_iteration == 10)]
    df = df[['Activationsparsity', 'FFweightsparsity', 'Numsegments', 'mean_accuracy']]
    # df.head()

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(14,10))

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])

    x1 = "Activationsparsity"
    x2 = "FFweightsparsity"
    x3 = "Numsegments"

    y = "mean_accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2
    fig.suptitle('Impact of the different hyperparameters on performance - \n \
                 full cross product of hyperparameters ', fontsize=12)

    pt.RainCloud(x = x1, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x3, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax3, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    ax2.set_ylabel('')
    ax2.set(yticklabels=[])
    ax3.set_ylabel('')
    ax3.set(yticklabels=[])

    if savefigs:
        plt.savefig(f'{figs_dir}/first_iteration_figure1.png', bbox_inches='tight')


    # Figure 2
    # Impact of each hyperparameter given other hyperparamers fixed to their best
    # value for best accuracy
    df2 = df[(df.Activationsparsity == 0.1) & (df.FFweightsparsity == 0.5)]
    df3 = df[(df.Activationsparsity == 0.1) & (df.Numsegments == 10)]
    df4 = df[(df.FFweightsparsity == 0.5) & (df.Numsegments == 10)]


    gs = gridspec.GridSpec(1, 3)
    fig2 = plt.figure(figsize=(14,10))

    ax1 = fig2.add_subplot(gs[:, 0])
    ax2 = fig2.add_subplot(gs[:, 1])
    ax3 = fig2.add_subplot(gs[:, 2])

    x1 = "Numsegments"
    x2 = "FFweightsparsity"
    x3 = "Activationsparsity"

    y = "mean_accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2

    fig2.suptitle('Impact of each hyperparameter given other hyperparamers fixed \n \
                  to their best value for best accuracy', fontsize=16)

    pt.RainCloud(x = x1, y = y, data = df2, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df3, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x3, y = y, data = df4, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax3, orient = ort, move = .2,
                     pointplot = True, alpha = .65)

    if savefigs:
        plt.savefig(f'{figs_dir}/first_iteration_figure2.png', bbox_inches='tight')

def second_iteration():
    '''
    Second experimental runs. Sadly some error aborted earlier than expected and
    missing data. Contrary to the first iteration this uses Subutai's script to
    aggregate data and not my own.
    '''

    experiment_folder = '/home/jeremy/nta/results/experiments/dendrites/cns_abstract_2021_2/'
    df_path = f'{experiment_folder}temp.csv'
    df = pd.read_csv(df_path)

    savefigs = True
    figs_dir = 'figs/'
    if savefigs:
        if not os.path.isdir(f'{figs_dir}'):
            os.makedirs(f'{figs_dir}')

    df.columns
    df = df[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(14,10))

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])

    x1 = "Activation sparsity"
    x2 = "FF weight sparsity"
    x3 = "Num segments"

    y = "Accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2
    fig.suptitle('Impact of the different hyperparameters on performance - \n \
                 full cross product of hyperparameters ', fontsize=12)

    pt.RainCloud(x = x1, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x3, y = y, data = df, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax3, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    ax2.set_ylabel('')
    ax2.set(yticklabels=[])
    ax3.set_ylabel('')
    ax3.set(yticklabels=[])

    if savefigs:
        plt.savefig(f'{figs_dir}/second_iteration_figure1.png', bbox_inches='tight')


    # Figure 2
    # Impact of each hyperparameter given other hyperparamers fixed to their best
    # value for best accuracy
    df2 = df[(df['Activation sparsity'] == 0.1) & (df["FF weight sparsity"] == 0.5)]
    df3 = df[(df['Activation sparsity'] == 0.1) & (df['Num segments'] == 10)]
    df4 = df[(df["FF weight sparsity"] == 0.5) & (df['Num segments'] == 10)]


    gs = gridspec.GridSpec(1, 3)
    fig2 = plt.figure(figsize=(14,10))

    ax1 = fig2.add_subplot(gs[:, 0])
    ax2 = fig2.add_subplot(gs[:, 1])
    ax3 = fig2.add_subplot(gs[:, 2])

    x1 = "Num segments"
    x2 = "FF weight sparsity"
    x3 = "Activation sparsity"

    y = "Accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2

    fig2.suptitle('Impact of each hyperparameter given other hyperparamers fixed \n \
                  to their best value for best accuracy', fontsize=16)

    pt.RainCloud(x = x1, y = y, data = df2, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df3, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x3, y = y, data = df4, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax3, orient = ort, move = .2,
                     pointplot = True, alpha = .65)

    if savefigs:
        plt.savefig(f'{figs_dir}/second_iteration_figure2.png', bbox_inches='tight')

def third_iteration():
    '''
    Third experimental runs. Technically this is 3 separate runs as I only ran some
    of the hyperparameters, variying only one at the time and using Karan's
    CENTROID method ones for the others.
    Refer to configs:
        nb_segment_search2 = NB_SEGMENT_SEARCH_2,
        nb_segment_search3 = NB_SEGMENT_SEARCH_3,
        kw_sparsity_search = KW_SPARSITY_SEARCH,
        weights_sparsity_search = W_SPARSITY_SEARCH,
        weights_sparsity_search2 = W_SPARSITY_SEARCH2
    '''

    savefigs = True
    figs_dir = 'figs/'
    if savefigs:
        if not os.path.isdir(f'{figs_dir}'):
            os.makedirs(f'{figs_dir}')

    experiment_folder1 = '~/nta/results/experiments/dendrites/nb_segment_search2/'
    df_path1 = f'{experiment_folder1}temp.csv'
    df1 = pd.read_csv(df_path1)

    experiment_folder1bis = '~/nta/results/experiments/dendrites/nb_segment_search3/'
    df_path1bis = f'{experiment_folder1bis}temp.csv'
    df1bis = pd.read_csv(df_path1bis)

    # experiment_folder1bis2 = '~/nta/results/experiments/dendrites/nb_segment_search4/'
    # df_path1bis2 = f'{experiment_folder1bis2}temp.csv'
    # df1bis2 = pd.read_csv(df_path1bis2)

    experiment_folder2 = '~/nta/results/experiments/dendrites/kw_sparsity_search/'
    df_path2 = f'{experiment_folder2}temp.csv'
    df2 = pd.read_csv(df_path2)

    experiment_folder3 = '~/nta/results/experiments/dendrites/weights_sparsity_search/'
    df_path3 = f'{experiment_folder3}temp.csv'
    df3 = pd.read_csv(df_path3)

    experiment_folder3bis = '~/nta/results/experiments/dendrites/weights_sparsity_search2/'
    df_path3bis = f'{experiment_folder3bis}temp.csv'
    df3bis = pd.read_csv(df_path3bis)

    df1 = df1[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df1bis = df1bis[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    # df1bis2 = df1bis2[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df2 = df2[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df3 = df3[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df3bis = df3bis[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]

    df1 = pd.concat([df1, df1bis])
    # df1 = pd.concat([df1, df1bis2])
    df3 = pd.concat([df3, df3bis])

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(14,10))

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])

    x1 = "Num segments"
    x2 = "Activation sparsity"
    x3 = "FF weight sparsity"

    y = "Accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2
    fig.suptitle('Impact of the different hyperparameters on performance - \n \
                 full cross product of hyperparameters ', fontsize=12)

    pt.RainCloud(x = x1, y = y, data = df1, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df2, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x3, y = y, data = df3, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax3, orient = ort, move = .2,
                     pointplot = True, alpha = .65)
    ax1.set(xlabel = 'Number of dendritic segments')
    ax2.set(ylabel = '')
    ax2.set(xticklabels = ['0.99', '0.95', '0.9', '0.8', '0.6', '0.4', '0.2',
                        '0.1'])
    ax3.set(ylabel = '')
    ax3.set(xticklabels = ['0.99', '0.95', '0.9', '0.5', '0.3', '0.1', '0.05',
                        '0.01'])


    if savefigs:
        plt.savefig(f'{figs_dir}/third_iteration_figure1.png', bbox_inches='tight')

def cns_figure_1c():
    '''
    CNS 2021 abstract figure 1C. Using only part of the third iteration data
    and a bit more cleaned up
    This has been moved to nta/.../dendrites/permutedMNIST/experiments/figure1c.py
    and this function here might not be up to date.
    '''

    savefigs = True
    figs_dir = 'figs/'
    if savefigs:
        if not os.path.isdir(f'{figs_dir}'):
            os.makedirs(f'{figs_dir}')

    experiment_folder1 = '/home/jeremy/nta/results/experiments/dendrites/nb_segment_search2/'
    df_path1 = f'{experiment_folder1}temp.csv'
    df1 = pd.read_csv(df_path1)

    experiment_folder1bis = '/home/jeremy/nta/results/experiments/dendrites/nb_segment_search3/'
    df_path1bis = f'{experiment_folder1bis}temp.csv'
    df1bis = pd.read_csv(df_path1bis)

    experiment_folder2 = '/home/jeremy/nta/results/experiments/dendrites/kw_sparsity_search/'
    df_path2 = f'{experiment_folder2}temp.csv'
    df2 = pd.read_csv(df_path2)

    df1 = df1[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df1bis = df1bis[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]
    df2 = df2[['Activation sparsity', 'FF weight sparsity', 'Num segments', 'Accuracy']]

    df1 = pd.concat([df1, df1bis])

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(10,8))

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    x1 = "Num segments"
    x2 = "Activation sparsity"

    y = "Accuracy"
    ort = "v"
    pal = "Set2"
    sigma = .2
    fig.suptitle('Impact of the number of dendritic segments or the \n \
                 activation sparsity on 10-tasks permuted MNIST performance',
                 fontsize=12)

    pt.RainCloud(x = x1, y = y, data = df1, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax1, orient = ort, move = .3,
                     pointplot = True, alpha = .65)
    pt.RainCloud(x = x2, y = y, data = df2, palette = pal, bw = sigma,
                     width_viol = .6, ax = ax2, orient = ort, move = .3,
                     pointplot = True, alpha = .65)
    ax1.set_ylim([.68, .98])
    ax1.set_ylabel('Mean accuracy')
    ax1.set_xlabel('Number of dendritic segments (n)')
    # for i in range(len(ax1.lines)):
    #     ax1.lines[i].set_linestyle('--') #not sure which one should be referenced here
    #     ax1.lines[i].set_linewidth(1)
    #     ax1.lines[i].scale = 10
    ax2.set_ylim([.68, .98])
    ax2.set_ylabel('')
    ax2.set(yticklabels=[])
    ax2.set_xlabel('Activation sparsity (k-winner %)')


    if savefigs:
        plt.savefig(f'{figs_dir}/cns2021_figure1c.png', bbox_inches='tight',
                    dpi=1200)

if __name__ == '__main__':
    # first_iteration()
    # second_iteration()
    third_iteration()
    # cns_figure_1c()
