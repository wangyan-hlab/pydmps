import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sns.set('talk', 'whitegrid', 'bright', font_scale=0.7,
        rc={"lines.linewidth": 1.5, 'grid.linestyle': '--'})

import csv
import utiltools.robotmath as rm
import pandaplotutils.pandageom as pg

t, px, py, pz, prx, pry, prz, trajectory = ([] for i in range(8))

with open('./demopath_Lsmall/L_small_path.csv', 'r', newline='') as demopathLsmall:
    pose = list(csv.reader(demopathLsmall))

for p in pose:
    traj = list(map(lambda k: eval(k), p[:6]))
    px.append(eval(p[0]))
    py.append(eval(p[1]))
    pz.append(eval(p[2]))
    prx.append(eval(p[3]))
    pry.append(eval(p[4]))
    prz.append(eval(p[5]))
    trajectory.append(traj)

x_, y_, z_, roll_, pitch_, yaw_ = ([] for i in range(6))
for relpose in trajectory:
    x_.append(relpose[0])
    z_.append(relpose[2])
    pitch_.append(relpose[4])

x_des = np.array([x_])
np.savez('../data/demopath_Lsmall_x.npz', x_des.T)
z_des = np.array([z_])
np.savez('../data/demopath_Lsmall_z.npz', z_des.T)
pitch_des = np.array([pitch_])
np.savez('../data/demopath_Lsmall_pitch.npz', pitch_des.T)

plt.figure(1, figsize=(8, 8))
plt.subplot(311)
plt.title('x, z, and pitch of the demopath_Lsmall')
# plt.xticks([])
# plt.ylim((-105,-41))
plt.xlabel('steps')
plt.ylabel('x(mm)')
plt.plot(x_, 'r-', label='x')
plt.legend()

plt.subplot(312)
# plt.xticks([])
# plt.ylim((25, 85))
plt.xlabel('steps')
plt.ylabel('z(mm)')
plt.plot(z_, 'b-', label='z')
plt.legend()

plt.subplot(313)
# plt.ylim((-5,25))
plt.xlabel('steps')
plt.ylabel('pitch angle($^\circ$)')
plt.plot(pitch_, 'g-', label='pitch')
plt.legend()
plt.show()
