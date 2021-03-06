"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import pydmps
import pydmps.dmp_discrete_modified as dmpmod

y_des = np.load("2.npz")["arr_0"].T
# y_des = np.load("../yan/5.npz")["arr_0"].T
y_des -= y_des[:, 0][:, None]

# test normal run
k = 10
n_dmps = 2
ay = np.ones(n_dmps) * k
dmp = dmpmod.DMPs_discrete_modified(n_dmps=n_dmps, n_bfs=500, ay=ay)

dmp.imitate_path(y_des=y_des)

plt.figure(1, figsize=(6, 6))

y_track, dy_track, ddy_track = dmp.rollout()
plt.plot(y_track[:, 0], y_track[:, 1], "b--", lw=2, alpha=0.5)

# run while moving the target up and to the right
y_track = []
dmp.reset_state()
# print('dd', dmp.timesteps)
for t in range(dmp.timesteps):
    y, _, _ = dmp.step()
    y_track.append(np.copy(y))
    # move the target slightly every time step
    dmp.goal += np.array([0.005, -0.005])
    print('gggggggggg', dmp.goal)
y_track = np.array(y_track)

plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.xlim([int(min(y_des[0]))-1, int(max(y_des[0]))+1])
plt.ylim([int(min(y_des[1]))-1, int(max(y_des[1]))+1])
plt.legend(["original path", "moving target"])
plt.grid()
plt.show()
