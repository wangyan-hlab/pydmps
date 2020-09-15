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
obj = 'L'
# obj = 'L_small'
x_des = []
y_des = []
pitch_des = []
if obj == 'L':
    x_des = np.load("../data/demopath_x.npz")["arr_0"].T[1]
    z_des = np.load("../data/demopath_z.npz")["arr_0"].T[1]
    pitch_des = np.load("../data/demopath_pitch.npz")["arr_0"].T[1]
elif obj == 'L_small':
    x_des = np.load("../data/demopath_Lsmall_x.npz")["arr_0"].T[0]
    z_des = np.load("../data/demopath_Lsmall_z.npz")["arr_0"].T[0]
    pitch_des = np.load("../data/demopath_Lsmall_pitch.npz")["arr_0"].T[0]

x_mov = np.load("../data/demopath_Lsmall_x.npz")["arr_0"].T[0]
z_mov = np.load("../data/demopath_Lsmall_z.npz")["arr_0"].T[0]
pitch_mov = np.load("../data/demopath_Lsmall_pitch.npz")["arr_0"].T[0]
timesteps_mov = len(x_mov)

dt = 1/len(x_des)
tau1 = 1.0
tau2 = 1.0
tau3 = 1.0
# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(dt=dt, n_dmps=1, n_bfs=1000, ay=np.ones(1) * 25.0)
dmp.imitate_path(y_des=x_des)

# print("DMP_x - weights of the foring term BFs:", dmp.w)
plt.figure(1, figsize=(8, 8))
plt.subplot(311)
x_track, dx_track, ddx_track = dmp.rollout()
plt.plot(x_track, "r--", alpha=0.5, label="original x")

# run while moving the target up and to the right
x_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    # move the target
    if t <= int(dmp.timesteps/3):
        dmp.y0 = np.asarray([x_des[0]])
        dmp.goal = np.array([x_mov[int(timesteps_mov/3)]])
        y, _, _ = dmp.step(tau=tau1)
        x_track.append(np.copy(y))
    elif int(dmp.timesteps/3) < t <= int(dmp.timesteps*3/4):
        dmp.y0 = np.array([x_mov[int(timesteps_mov/3)]])
        dmp.goal = np.array([x_mov[int(timesteps_mov*3/4)]])
        y, _, _ = dmp.step(tau=tau2)
        x_track.append(np.copy(y))
        # print("goal =", dmp.goal)
    elif t > int(dmp.timesteps*3/4):
        dmp.y0 = np.array([x_mov[int(timesteps_mov*3/4)]])
        dmp.goal = np.array([x_mov[-1]])
        y, _, _ = dmp.step(tau=tau3)
        x_track.append(np.copy(y))

x_track = np.array(x_track)
plt.plot(x_track, "r", label="moving x")
# plt.title("DMP system - follow demonstration path")
plt.title("DMP system")

# plt.axis("equal")
plt.xlabel('steps')
plt.ylabel('x(mm)')
# plt.ylim(-105, -35)
plt.legend(ncol=1, loc="lower right", shadow=True, borderpad=.5)

# ----------------------------------------------------------------------------------

dmp = pydmps.dmp_discrete.DMPs_discrete(dt=dt, n_dmps=1, n_bfs=1000, ay=np.ones(1) * 25.0)
dmp.imitate_path(y_des=z_des)

plt.subplot(312)
z_track, dz_track, ddz_track = dmp.rollout()
plt.plot(z_track, "b--", alpha=0.5, label="original z")

# run while moving the target up and to the right
z_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    # move the target
    if t <= int(dmp.timesteps/3):
        dmp.y0 = np.asarray([z_des[0]])
        dmp.goal = np.array([z_mov[int(timesteps_mov/3)]])
        y, _, _ = dmp.step(tau=tau1)
        z_track.append(np.copy(y))
    elif int(dmp.timesteps/3) < t <= int(dmp.timesteps*3/4):
        dmp.y0 = np.array([z_mov[int(timesteps_mov/3)]])
        dmp.goal = np.array([z_mov[int(timesteps_mov*3/4)]])
        y, _, _ = dmp.step(tau=tau2)
        z_track.append(np.copy(y))
        # print("goal =", dmp.goal)
    elif t > int(dmp.timesteps*3/4):
        dmp.y0 = np.array([z_mov[int(timesteps_mov*3/4)]])
        dmp.goal = np.array([z_mov[-1]])
        y, _, _ = dmp.step(tau=tau3)
        z_track.append(np.copy(y))

z_track = np.array(z_track)
plt.plot(z_track, "b", label="moving z")

# plt.axis("equal")
plt.xlabel('steps')
plt.ylabel('z(mm)')
# plt.ylim(18, 85)
plt.legend(ncol=1, shadow=True, borderpad=.5)

# ----------------------------------------------------------------------------------

dmp = pydmps.dmp_discrete.DMPs_discrete(dt=dt, n_dmps=1, n_bfs=1000, ay=np.ones(1) * 25.0)
dmp.imitate_path(y_des=pitch_des)

plt.subplot(313)
pitch_track, dpitch_track, ddpitch_track = dmp.rollout()
plt.plot(pitch_track, "g--", alpha=0.5, label="original P")

# run while moving the target up and to the right
pitch_track = []
dmp.reset_state()
for t in range(dmp.timesteps):
    if t <= int(dmp.timesteps/3):
        dmp.y0 = np.asarray([pitch_des[0]])
        dmp.goal = np.array([pitch_mov[int(timesteps_mov/3)]])
        y, _, _ = dmp.step(tau=tau1)
        pitch_track.append(np.copy(y))
    elif int(dmp.timesteps/3) < t <= int(dmp.timesteps*3/4):
        dmp.y0 = np.array([pitch_mov[int(timesteps_mov/3)]])
        dmp.goal = np.array([pitch_mov[int(timesteps_mov*3/4)]])
        y, _, _ = dmp.step(tau=tau2)
        pitch_track.append(np.copy(y))
        # print("goal =", dmp.goal)
    elif t > int(dmp.timesteps*3/4):
        dmp.y0 = np.array([pitch_mov[int(timesteps_mov*3/4)]])
        dmp.goal = np.array([pitch_mov[-1]])
        y, _, _ = dmp.step(tau=tau3)
        pitch_track.append(np.copy(y))

pitch_track = np.array(pitch_track)
plt.plot(pitch_track, "g", label="moving P")

# plt.axis("equal")
plt.xlabel('steps')
plt.ylabel('pitch angle($^\circ$)')
# plt.ylim(-5, 25)
plt.legend(ncol=1, shadow=True, borderpad=.5)

plt.show()
