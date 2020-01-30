# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
from importlib import import_module

import numpy as np
from ray import tune

from nupic.research.frameworks.backprop_structure.networks import (
    gsc_lenet_backpropstructure,
    gsc_lesparsenet,
)
from nupic.research.frameworks.backprop_structure.ray_ax import (
    get_best_config,
    get_frontier_trials,
)


def product(xs):
    y = 1.
    for x in xs:
        y *= x
    return y


def get_densenet_weights_by_layer(densenet_constructor):
    lesparsenet = densenet_constructor()
    return {
        name1: getattr(lesparsenet, name2).weight.numel()
        for name1, name2 in [("cnn1", "cnn1_cnn"),
                             ("cnn2", "cnn2_cnn"),
                             ("fc1", "linear1_linear"),
                             ("fc2", "output")]
    }


def get_hsd_weights_by_layer(hsd_constructor):
    lesparsenet = hsd_constructor()
    hsd_weights_by_layer = {
        "fc2": lesparsenet.output.weight.numel(),
    }
    for name1, name2 in [("cnn1", "cnn1_cnn"),
                         ("cnn2", "cnn2_cnn"),
                         ("fc1", "linear1")]:
        m = getattr(lesparsenet, name2)
        hsd_weights_by_layer[name1] = (m.weight_sparsity
                                       * m.module.weight.numel())
    return hsd_weights_by_layer


def weight_table_row(label, acc,
                     cnn1_nz, cnn1_wmatrix_size,
                     cnn2_nz, cnn2_wmatrix_size,
                     fc1_nz, fc1_wmatrix_size,
                     fc2_nz, fc2_wmatrix_size):
    cnn1_p = cnn1_nz / product(cnn1_wmatrix_size)
    cnn1_pu = cnn1_nz / cnn1_wmatrix_size[0]
    cnn2_p = cnn2_nz / product(cnn2_wmatrix_size)
    cnn2_pu = cnn2_nz / cnn2_wmatrix_size[0]
    fc1_p = fc1_nz / product(fc1_wmatrix_size)
    fc1_pu = fc1_nz / fc1_wmatrix_size[0]
    fc2_p = fc2_nz / product(fc2_wmatrix_size)
    fc2_pu = fc2_nz / fc2_wmatrix_size[0]

    tot_nz = cnn1_nz + cnn2_nz + fc1_nz + fc2_nz
    tot_p = tot_nz / (product(cnn1_wmatrix_size) + product(cnn2_wmatrix_size)
                      + product(fc1_wmatrix_size) + product(fc2_wmatrix_size))

    return f"""
\\multirow{{3}}{{7.95em}}{{\\textbf{{{label}}} \\\\ {acc:.2%} accuracy}}
  & \\textit{{Percent}} & {cnn1_p:.0%} & {cnn2_p:.0%} &
                          {fc1_p:.0%} & {fc2_p:.0%} & {tot_p:.0%} \\\\
  & \\textit{{Total}} & {cnn1_nz:,.0f} & {cnn2_nz:,.0f} &
                        {fc1_nz:,.0f} & {fc2_nz:,.0f} & \\textbf{{{tot_nz:,.0f}}} \\\\
  & \\textit{{Per unit}} & {cnn1_pu:,.0f} & {cnn2_pu:,.0f} &
                           {fc1_pu:,.0f} & {fc2_pu:,.0f} & \\\\
    """.replace("%", "\\%")


def save_weight_table(all_bps_results, bps_constructor,
                      hsd_results, hsd_constructor,
                      dense_results, densenet_constructor):

    hsd_weights_by_layer = get_hsd_weights_by_layer(hsd_constructor)
    dense_weights_by_layer = get_densenet_weights_by_layer(densenet)
    lesparsenet = hsd_constructor()
    lenet_bps = bps_constructor()
    dense = densenet_constructor()

    rows = [
        weight_table_row(
            "Dense",
            np.mean([result["mean_accuracy"]
                     for result in dense_results]),

            dense_weights_by_layer["cnn1"],
            dense.cnn1_cnn.weight.size(),
            dense_weights_by_layer["cnn2"],
            dense.cnn2_cnn.weight.size(),
            dense_weights_by_layer["fc1"],
            dense.linear1_linear.weight.size(),
            dense_weights_by_layer["fc2"],
            dense.output.weight.size(),
        ),
    ]

    for bps_results in all_bps_results:
        rows.append(weight_table_row(
            "Dynamic sparse",
            np.mean([result["mean_accuracy"]
                     for result in bps_results]),

            np.mean([result["cnn1/inference_nz"]
                     for result in bps_results]),
            lenet_bps.cnn1.weight_size(),

            np.mean([result["cnn2/inference_nz"]
                     for result in bps_results]),
            lenet_bps.cnn2.weight_size(),

            np.mean([result["fc1/inference_nz"]
                     for result in bps_results]),
            lenet_bps.fc1.weight_size(),

            np.mean([result["fc2/inference_nz"]
                     for result in bps_results]),
            lenet_bps.fc2.weight_size()))

    rows.append(weight_table_row(
        "Static sparse",
        np.mean([result["mean_accuracy"]
                 for result in hsd_results]),

        hsd_weights_by_layer["cnn1"],
        lesparsenet.cnn1_cnn.module.weight.size(),
        hsd_weights_by_layer["cnn2"],
        lesparsenet.cnn2_cnn.module.weight.size(),
        hsd_weights_by_layer["fc1"],
        lesparsenet.linear1.module.weight.size(),
        hsd_weights_by_layer["fc2"],
        lesparsenet.output.weight.size(),
    ))

    weight_table = """
\\begin{{table}}
\\begin{{adjustwidth}}{{-1in}}{{-1in}}
\\begin{{tabular}}{{cc|ccccc}}
\textit{\textbf{Weights}} & & Conv1 & Conv2 & FC1 & FC2 & \\textbf{{Total}} \\\\
\\hline
{}
\\end{{tabular}}
\\end{{adjustwidth}}
\\end{{table}}
    """.format("\\hline".join(rows))

    print(weight_table)


def mult_table_row(label, acc, mults):
    cnn1 = mults["cnn1"]
    cnn2 = mults["cnn2"]
    fc1 = mults["fc1"]
    fc2 = mults["fc2"]
    tot = sum(mults.values())

    return f"""
\\multirow{{2}}{{7.95em}}{{\\textbf{{{label}}} \\\\ {acc:.2%} accuracy}}
    & \\multirow{{2}}{{4em}}{{\\raggedleft {cnn1:,.0f}}}
    & \\multirow{{2}}{{4.5em}}{{\\raggedleft {cnn2:,.0f}}}
    & \\multirow{{2}}{{4em}}{{\\raggedleft {fc1:,.0f}}}
    & \\multirow{{2}}{{3em}}{{\\raggedleft {fc2:,.0f}}}
    & \\multirow{{2}}{{4.5em}}{{\\raggedleft \\textbf{{{tot:,.0f}}}}} \\\\
    \\\\
    """.replace("%", "\\%")


def save_multiplies_table(all_bps_results,
                          hsd_results, hsd_constructor,
                          dense_results, densenet_constructor):
    # Use the BPS results to infer how many times each filter is applied.
    filter_applications_by_layer = {
        layername: (all_bps_results[0][0][f"{layername}/multiplies"]
                    / all_bps_results[0][0][f"{layername}/inference_nz"])
        for layername in ["cnn1", "cnn2", "fc1", "fc2"]
    }

    hsd_weights_by_layer = get_hsd_weights_by_layer(hsd_constructor)
    hsd_mults = {
        name: (hsd_weights_by_layer[name]
               * filter_applications_by_layer[name])
        for name in ["cnn1", "cnn2", "fc1", "fc2"]
    }

    densenet_weights_by_layer = get_densenet_weights_by_layer(densenet_constructor)
    densenet_mults = {
        name: (densenet_weights_by_layer[name]
               * filter_applications_by_layer[name])
        for name in ["cnn1", "cnn2", "fc1", "fc2"]
    }

    rows = [
        mult_table_row(
            "Dense",
            np.mean([result["mean_accuracy"]
                     for result in dense_results]),
            densenet_mults
        ),
    ]

    for bps_results in all_bps_results:
        bps_mults = {
            name: np.mean([result[f"{name}/multiplies"]
                           for result in bps_results])
            for name in ["cnn1", "cnn2", "fc1", "fc2"]
        }

        rows.append(mult_table_row(
            "Dynamic sparse",
            np.mean([result["mean_accuracy"]
                     for result in bps_results]),
            bps_mults
        ))

    rows.append(mult_table_row(
        "Static sparse",
        np.mean([result["mean_accuracy"]
                 for result in hsd_results]),
        hsd_mults
    ))

    mult_table = """
\\begin{{table}}
\\begin{{adjustwidth}}{{-1in}}{{-1in}}
\\begin{{tabular}}{{c|ccccc}}
\textit{\textbf{Multiplies}} & Conv1 & Conv2 & FC1 & FC2 & \\textbf{{Total}} \\\\
\\hline
{}
\\end{{tabular}}
\\end{{adjustwidth}}
\\end{{table}}
    """.format("\\hline".join(rows))

    print(mult_table)


def densenet():
    return gsc_lesparsenet(
        cnn_activity_percent_on=(1.0, 1.0),
        cnn_weight_percent_on=(1.0, 1.0),
        linear_activity_percent_on=(1.0,),
        linear_weight_percent_on=(1.0,)
    )


def save_tables():
    hsd_exp = "run_lenet_staticstructure_gsc"
    m = import_module(f"runs.{hsd_exp}")
    df = tune.Analysis(os.path.expanduser(f"~/ray_results/{hsd_exp}")).dataframe()
    df = df[df["training_iteration"] == m.NUM_TRAINING_ITERATIONS]
    hsd_results = [result
                   for _, result in df.iterrows()]
    hsd_acc = np.mean([result["mean_accuracy"] for result in hsd_results])

    dense_exp = "ax_ln_gsc"
    m = import_module(f"runs.{dense_exp}")
    _, dense_results = get_best_config(
        os.path.expanduser(f"~/ray_results/{dense_exp}"),
        m.PARAMETERS,
        m.NUM_TRAINING_ITERATIONS)

    exp_name = "ax_ln_bps_alt_gsc"
    m = import_module(f"runs.{exp_name}")

    frontier_trials = get_frontier_trials(
        os.path.expanduser(f"~/ray_results/{exp_name}"),
        m.PARAMETERS,
        m.NUM_TRAINING_ITERATIONS)
    accs = np.array([np.mean([result["mean_accuracy"]
                              for result in results])
                     for _, results in frontier_trials])

    diffs = accs - hsd_acc
    diffs[diffs < 0] = np.inf
    bps_config1, bps_results1 = frontier_trials[diffs.argmin()]

    # Ideally this would use the largest accuracy that achieves some minimal
    # sparsity requirement. This just happens to work using my current results.
    # (Yes, it's a hack.)
    bps_config2, bps_results2 = frontier_trials[accs.argmax()]

    save_weight_table([bps_results2, bps_results1], gsc_lenet_backpropstructure,
                      hsd_results, gsc_lesparsenet,
                      dense_results, densenet)

    save_multiplies_table([bps_results2, bps_results1], hsd_results, gsc_lesparsenet,
                          dense_results, densenet)


if __name__ == "__main__":
    save_tables()
