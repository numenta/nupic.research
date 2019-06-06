# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# This uses plotly to create a nice looking graph of average false positive
# error rates as a function of N, the dimensionality of the vectors.  I'm sorry
# this code is so ugly.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt # noqa E402 I001

# Observed vs theoretical error values

# a=64 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a64 = [1.09318E-03, 5.74000E-06, 1.10000E-07]

theoretical_errors_a64 = [0.00109461662333690, 5.69571108769533e-6,
                          1.41253230930730e-7]

# a=128 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a128 = [0.292048, 0.00737836, 0.00032014, 0.00002585,
                            0.00000295, 0.00000059, 0.00000013, 0.00000001,
                            0.00000001]

theoretical_errors_a128 = [0.292078213737764, 0.00736788303358289,
                           0.000320106080889471, 2.50255519815378e-5,
                           2.99642102590114e-6,
                           4.89399786076359e-7, 1.00958512780931e-7,
                           2.49639031779358e-8,
                           7.13143762262004e-9]

# a=256 cells active, s=24 synapses on segment, dendritic threshold is theta=12
experimental_errors_a256 = [
    9.97368E-01, 6.29267E-01, 1.21048E-01, 1.93688E-02, 3.50879E-03,
    7.49560E-04,
    1.86590E-04, 5.33200E-05, 1.65000E-05, 5.58000E-06, 2.23000E-06,
    9.30000E-07,
    3.20000E-07, 2.70000E-07, 7.00000E-08, 4.00000E-08, 2.00000E-08
]

# a=n/2 cells active, s=24 synapses on segment, dendritic threshold is theta=12
errors_dense = [0.584014929308308, 0.582594747080399, 0.582007206016863,
                0.581686021979051, 0.581483533877904, 0.581344204898149,
                0.581242471033283,
                0.581164924569868, 0.581103856001899, 0.581054517612207,
                0.581013825794851,
                0.580979690688467, 0.580950645707841, 0.580925631309445,
                0.580903862938630,
                0.580884747253428, 0.580867827216677]

theoretical_errors_a256 = [0.999997973443107, 0.629372754740777,
                           0.121087724790945, 0.0193597645959856,
                           0.00350549721741729,
                           0.000748965962032781, 0.000186510373919969,
                           5.30069204544174e-5,
                           1.68542688790000e-5, 5.89560747849969e-6,
                           2.23767020178735e-6,
                           9.11225564771580e-7, 3.94475072403605e-7,
                           1.80169987461924e-7,
                           8.62734957588259e-8, 4.30835081022293e-8,
                           2.23380881095835e-8]

list_of_n_values = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100,
                    2300,
                    2500, 2700, 2900, 3100, 3300, 3500]

fig, ax = plt.subplots()

fig.suptitle("Match probability for sparse binary vectors")
ax.set_xlabel("Dimensionality (n)")
ax.set_ylabel("Frequency of matches")
ax.set_yscale("log")

ax.scatter(list_of_n_values[0:3], experimental_errors_a64,
           label="a=64 (predicted)",
           marker="o", color='black')
ax.scatter(list_of_n_values[0:9], experimental_errors_a128,
           label="a=128 (predicted)", marker="o", color='black')
ax.scatter(list_of_n_values, experimental_errors_a256,
           label="a=256 (predicted)",
           marker="o", color='black')

ax.plot(list_of_n_values, errors_dense, 'k:', label="a=n/2 (predicted)",
        color='black')

ax.plot(list_of_n_values[0:3], theoretical_errors_a64, 'k:',
        label="a=64 (observed)")
ax.plot(list_of_n_values[0:9], theoretical_errors_a128, 'k:',
        label="a=128 (observed)", color='black')
ax.plot(list_of_n_values, theoretical_errors_a256, 'k:',
        label="a=256 (observed)")

ax.annotate(r"$a = 64$", xy=(list_of_n_values[2], theoretical_errors_a64[-1]),
            xytext=(-5, 2), textcoords="offset points", ha="right",
            color='black')
ax.annotate(r"$a = 128$", xy=(list_of_n_values[8], theoretical_errors_a64[-1]),
            ha="center", color='black')
ax.annotate(r"$a = 256$", xy=(list_of_n_values[-1], theoretical_errors_a64[-1]),
            xytext=(-10, 0), textcoords="offset points", ha="center",
            color='black')
ax.annotate(r"$a = \frac{n}{2}$",
            xy=(list_of_n_values[-2], experimental_errors_a256[2]),
            xytext=(-10, 0), textcoords="offset points", ha="center",
            color='black')

plt.minorticks_off()
plt.grid(True, alpha=0.3)

plt.savefig("images/effect_of_n.pdf")
plt.close()
