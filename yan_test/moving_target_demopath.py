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
import pydmps
import pydmps.dmp_discrete
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set('talk', 'whitegrid', 'bright', font_scale=0.7,
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

x_des = np.load("../data/demopath_x.npz")["arr_0"].T
z_des = np.load("../data/demopath_z.npz")["arr_0"].T
pitch_des = np.load("../data/demopath_pitch.npz")["arr_0"].T

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 25.0)
dmp.imitate_path(y_des=x_des)

plt.figure(1, figsize=(8, 8))
plt.subplot(311)
x_track, dx_track, ddx_track = dmp.rollout()
plt.plot(x_track[:, 0], x_track[:, 1], "r--", alpha=0.5, label="original x")

# run while moving the target up and to the right
x_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    y, _, _ = dmp.step()
    x_track.append(np.copy(y))
    # move the target slightly every time step
    dmp.goal += np.array([0, 0.1])
x_track = np.array(x_track)

plt.plot(x_track[:, 0], x_track[:, 1], "r", label="moving x")
plt.title("DMP system - follow demonstration path")

# plt.axis("equal")
plt.xlabel('t(s)')
plt.ylabel('x(mm)')
plt.ylim(-105, -35)
plt.legend(ncol=1, loc="lower right", shadow=True, borderpad=.5)

# ----------------------------------------------------------------------------------

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 25.0)
dmp.imitate_path(y_des=z_des)

plt.subplot(312)
z_track, dz_track, ddz_track = dmp.rollout()
plt.plot(z_track[:, 0], z_track[:, 1], "b--", alpha=0.5, label="original z")

# run while moving the target up and to the right
z_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    z, _, _ = dmp.step()
    z_track.append(np.copy(z))
    # move the target slightly every time step
    dmp.goal += np.array([0, 0.1])
z_track = np.array(z_track)

plt.plot(z_track[:, 0], z_track[:, 1], "b", label="moving z")

# plt.axis("equal")
plt.xlabel('t(s)')
plt.ylabel('z(mm)')
plt.ylim(25, 85)
plt.legend(ncol=1, shadow=True, borderpad=.5)

# ----------------------------------------------------------------------------------

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 25.0)
dmp.imitate_path(y_des=pitch_des)

plt.subplot(313)
pitch_track, dpitch_track, ddpitch_track = dmp.rollout()
plt.plot(pitch_track[:, 0], pitch_track[:, 1], "g--", alpha=0.5, label="original P")

# run while moving the target up and to the right
pitch_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    pitch, _, _ = dmp.step()
    pitch_track.append(np.copy(pitch))
    # move the target slightly every time step
    dmp.goal += np.array([0, 0.1])
pitch_track = np.array(pitch_track)

plt.plot(pitch_track[:, 0], pitch_track[:, 1], "g", label="moving P")

# plt.axis("equal")
plt.xlabel('t(s)')
plt.ylabel('pitch angle($^\circ$)')
plt.ylim(-5, 25)
plt.legend(ncol=1, shadow=True, borderpad=.5)

plt.show()
