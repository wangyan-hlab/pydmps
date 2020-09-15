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

with open('./demopath_L/data20200226_151954.csv','r',newline='') as demopathL:
    pose = list(csv.reader(demopathL))

for p in pose[1:]:
    traj = list(map(lambda k: eval(k), p[:7]))
    t.append(eval(p[0]))
    px.append(eval(p[1]))
    py.append(eval(p[2]))
    pz.append(eval(p[3]))
    prx.append(eval(p[4]))
    pry.append(eval(p[5]))
    prz.append(eval(p[6]))
    trajectory.append(traj)

state = []
relstate = []
posG = np.array([710, 68, -310])
rotG = rm.rotmat_from_euler(0, 0, -4)

for i in range(len(trajectory)):
    rot_joint = rm.rotmat_from_euler(prx[i], pry[i], prz[i])
    pos_joint = np.array([px[i], py[i], pz[i]])
    joint_pose_to_base = rm.homobuild(pos_joint, rot_joint)
    rot_obj0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    pos_obj = np.array([0, 21, 54 + 15 + 130 + 15])
    obj_pose_to_joint0 = rm.homobuild(pos_obj, rot_obj0)
    obj_pose_to_joint = np.dot(rm.homobuild(np.array([0, 0, 0]), rm.rotmat_from_euler(0, 0, 225 + 22.5)),
                               obj_pose_to_joint0)
    obj_pose_to_base = np.dot(joint_pose_to_base, obj_pose_to_joint)
    p_obj = obj_pose_to_base[:3, 3]
    rpy_obj = rm.euler_from_matrix(pg.npToMat3(np.transpose(obj_pose_to_base[:3, :3])))
    state.append([p_obj[0], p_obj[1], p_obj[2], rpy_obj[0], rpy_obj[1], rpy_obj[2]])

    posL = np.array([p_obj[0], p_obj[1], p_obj[2]])
    rotL = rm.rotmat_from_euler(rpy_obj[0], rpy_obj[1], rpy_obj[2])
    relpos, relrot = rm.relpose(posG, rotG, posL, rotL)
    rot = rm.euler_from_matrix(relrot)
    relstate.append([round(relpos[0],3), round(relpos[1],3), round(relpos[2],3),
                     round(rot[0],3), round(rot[1],3), round(rot[2],3)])

x_, y_, z_, roll_, pitch_, yaw_ = ([] for i in range(6))
for relpose in relstate:
    x_.append(relpose[0])
    z_.append(relpose[2])
    pitch_.append(relpose[4])

# x_des = np.array([t, x_])
# np.savez('../data/demopath_x.npz', x_des.T)
# z_des = np.array([t, z_])
# np.savez('../data/demopath_z.npz', z_des.T)
# pitch_des = np.array([t, pitch_])
# np.savez('../data/demopath_pitch.npz', pitch_des.T)

plt.figure(1, figsize=(8, 8))
plt.subplot(311)
plt.title('x, z, and pitch of the demopath_L')
# plt.xticks([])
plt.ylim((-105,-41))
plt.xlabel('t(s)')
plt.ylabel('x(mm)')
plt.plot(t, x_, 'r-', label='x')
plt.legend()

plt.subplot(312)
# plt.xticks([])
plt.ylim((25, 85))
plt.xlabel('t(s)')
plt.ylabel('z(mm)')
plt.plot(t, z_, 'b-', label='z')
plt.legend()

plt.subplot(313)
plt.ylim((-5,25))
plt.xlabel('t(s)')
plt.ylabel('pitch angle($^\circ$)')
plt.plot(t, pitch_, 'g-', label='pitch')
plt.legend()
plt.show()
