# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
# This uses plotly to create a nice looking graph of average false positive
# error rates as a function of N, the dimensionality of the vectors.  I'm sorry
# this code is so ugly.

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Observed vs theoretical error values

# a=64 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a64 = [1.09318e-03, 5.74000e-06, 1.10000e-07]

theoretical_errors_a64 = [0.00109461662333690, 5.69571108769533e-6, 1.41253230930730e-7]


# a=128 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a128 = [
    0.292048,
    0.00737836,
    0.00032014,
    0.00002585,
    0.00000295,
    0.00000059,
    0.00000013,
    0.00000001,
    0.00000001,
]

theoretical_errors_a128 = [
    0.292078213737764,
    0.00736788303358289,
    0.000320106080889471,
    2.50255519815378e-5,
    2.99642102590114e-6,
    4.89399786076359e-7,
    1.00958512780931e-7,
    2.49639031779358e-8,
    7.13143762262004e-9,
]

# a=256 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a256 = [
    9.97368e-01,
    6.29267e-01,
    1.21048e-01,
    1.93688e-02,
    3.50879e-03,
    7.49560e-04,
    1.86590e-04,
    5.33200e-05,
    1.65000e-05,
    5.58000e-06,
    2.23000e-06,
    9.30000e-07,
    3.20000e-07,
    2.70000e-07,
    7.00000e-08,
    4.00000e-08,
    2.00000e-08,
]

# a=n/2 cells active, s=24 synapses on segment, dendritic threshold is theta=12
errors_dense = [
    0.00518604306750049,
    0.00595902789913702,
    0.00630387009654985,
    0.00649883841432922,
    0.00662414645898081,
    0.00671145554136860,
    0.00677576979476038,
    0.00682511455944402,
    0.00686417048273405,
    0.00689585128896232,
    0.00692206553525732,
    0.00694411560202313,
    0.00696292062841680,
    0.00697914780884254,
    0.00699329317658955,
    0.00700573317947932,
    0.00701675866709042,
]

theoretical_errors_a256 = [
    0.999997973443107,
    0.629372754740777,
    0.121087724790945,
    0.0193597645959856,
    0.00350549721741729,
    0.000748965962032781,
    0.000186510373919969,
    5.30069204544174e-5,
    1.68542688790000e-5,
    5.89560747849969e-6,
    2.23767020178735e-6,
    9.11225564771580e-7,
    3.94475072403605e-7,
    1.80169987461924e-7,
    8.62734957588259e-8,
    4.30835081022293e-8,
    2.23380881095835e-8,
]

listof_n_values = [
    300,
    500,
    700,
    900,
    1100,
    1300,
    1500,
    1700,
    1900,
    2100,
    2300,
    2500,
    2700,
    2900,
    3100,
    3300,
    3500,
]

fig, ax = plt.subplots()

fig.suptitle("Match probability for sparse binary vectors")
ax.set_xlabel("Dimensionality (n)")
ax.set_ylabel("Frequency of matches")
ax.set_yscale("log")

ax.scatter(
    listof_n_values[0:3],
    experimental_errors_a64,
    label="a=64 (predicted)",
    marker="o",
    color="black",
)
ax.scatter(
    listof_n_values[0:9],
    experimental_errors_a128,
    label="a=128 (predicted)",
    marker="o",
    color="black",
)
ax.scatter(
    listof_n_values,
    experimental_errors_a256,
    label="a=256 (predicted)",
    marker="o",
    color="black",
)

ax.plot(listof_n_values, errors_dense, "k:", label="a=n/2 (predicted)", color="black")

ax.plot(listof_n_values[0:3], theoretical_errors_a64, "k:", label="a=64 (observed)")
ax.plot(
    listof_n_values[0:9],
    theoretical_errors_a128,
    "k:",
    label="a=128 (observed)",
    color="black",
)
ax.plot(listof_n_values, theoretical_errors_a256, "k:", label="a=256 (observed)")

ax.annotate(
    r"$a = 64$",
    xy=(listof_n_values[2], theoretical_errors_a64[-1]),
    xytext=(-5, 2),
    textcoords="offset points",
    ha="right",
    color="black",
)
ax.annotate(
    r"$a = 128$",
    xy=(listof_n_values[8], theoretical_errors_a64[-1]),
    ha="center",
    color="black",
)
ax.annotate(
    r"$a = 256$",
    xy=(listof_n_values[-1], theoretical_errors_a64[-1]),
    xytext=(-10, 0),
    textcoords="offset points",
    ha="center",
    color="black",
)
ax.annotate(
    r"$a = \frac{n}{2}$",
    xy=(listof_n_values[-2], experimental_errors_a256[3]),
    xytext=(-10, 0),
    textcoords="offset points",
    ha="center",
    color="black",
)

plt.minorticks_off()
plt.grid(True, alpha=0.3)

plt.savefig("effect_of_n.pdf")
plt.close()
