import numpy as np
import pydmps.dmp_discrete_modified as dmp
import matplotlib.pyplot as plt
import utiltools.robotmath as rm
import copy
import pandas as pd
import seaborn as sns
sns.set()
sns.set('talk', 'whitegrid', 'bright', font_scale=0.5,
        rc={"lines.linewidth": 1.2, 'grid.linestyle': '--'})

if __name__ == '__main__':

    filename = "demopath_L/HGCIL_traj.csv"
    # filename = "demopath_L/RelState.csv"
    state = pd.read_csv(filename,
                        names=['obj_x', 'obj_y', 'obj_z', 'obj_R', 'obj_P', 'obj_Y'],
                        na_values="?", comment='\t', sep=",", skipinitialspace=True)
    position = []
    euler = []
    quaternion = []
    for index in range(len(state)):
    # for index in range(int(len(state)/100*0), int(len(state)/100*1)):
        x = state.at[index, 'obj_x']
        y = state.at[index, 'obj_y']
        z = state.at[index, 'obj_z']
        roll_deg = state.at[index, 'obj_R']
        pitch_deg = state.at[index, 'obj_P']
        yaw_deg = state.at[index, 'obj_Y']
        roll_rad = np.deg2rad(state.at[index, 'obj_R'])
        pitch_rad = np.deg2rad(state.at[index, 'obj_P'])
        yaw_rad = np.deg2rad(state.at[index, 'obj_Y'])
        position.append([x, y, z])
        euler.append([roll_deg, pitch_deg, yaw_deg])
        q = rm.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
        quaternion.append(q)

    position = np.asarray(position)
    euler = np.asarray(euler)
    quaternion = np.asarray(quaternion)
    y0_p_old = position[0]
    y0_euler_old = euler[0]
    y0_q_old = quaternion[0]
    g_p_old = position[-1]
    g_euler_old = euler[-1]
    g_q_old = quaternion[-1]
    print("y0_p_old: %s\ny0_euler_old: %s\ny0_q_old: %s\ng_p_old: %s\ng_euler_old: %s\ng_q_old: %s\n" %
          (y0_p_old, y0_euler_old, y0_q_old, g_p_old, g_euler_old, g_q_old))

    dt = 0.0002
    dmp_pos = dmp.DMPs_discrete_modified(dt=dt, n_dmps=3, n_bfs=100, ax=1.0, K=1000, dim='position')
    dmp_orn = dmp.DMPs_discrete_modified(dt=dt, n_dmps=4, n_bfs=100, ax=1.0, K=1000, dim='orientation')
    p_des = dmp_pos.imitate_path(y_des=position)
    q_des = dmp_orn.imitate_path(y_des=quaternion)

    dmp_pos.y0 = y0_p_old + np.array([-20, 0, 0])
    # print(dmp_pos.y0)
    dmp_orn.y0 = rm.quaternion_from_euler(np.deg2rad(y0_euler_old[0]+0),
                                          np.deg2rad(y0_euler_old[1]-5),
                                          np.deg2rad(y0_euler_old[2]+0))
    dmp_pos.goal = g_p_old + np.array([15, 0, -10])
    dmp_orn.goal = rm.quaternion_from_euler(np.deg2rad(g_euler_old[0]+0),
                                            np.deg2rad(g_euler_old[1]+4),
                                            np.deg2rad(g_euler_old[2]+0))
    # set up the DMP system - update start, goal, etc.
    dmp_pos.reset_state()
    dmp_orn.reset_state()

    p_track = []
    q_track = []

    flag_p = False
    t_p = 0
    while not flag_p:
    # for t in range(dmp_orn.timesteps):
        p, _, _ = dmp_pos.step(tau=1.0)
        p_track.append(copy.deepcopy(p))
        t_p += 1
        err_abs_p = np.linalg.norm(p - dmp_pos.goal)
        err_abs_p = np.nan_to_num(err_abs_p)
        err_rel_p = err_abs_p / (np.linalg.norm(dmp_pos.goal - dmp_pos.y0) + 1e-14)
        err_rel_p = np.nan_to_num(err_rel_p)
        if t_p % 1000 == 0:
            print("t_p =", t_p, "err_abs_p =", err_abs_p, "err_rel_p =", err_rel_p)
        flag_p = ((t_p >= dmp_orn.cs.timesteps) or err_rel_p <= 0.1)
    print("t_p = ", t_p)

    flag_q = False
    t_q = 0
    while not flag_q:
    # for t in range(dmp_orn.timesteps):
        q, _, _ = dmp_orn.step(tau=1.0)
        q_track.append(copy.deepcopy(q))
        t_q += 1
        err_abs_q = np.linalg.norm(q - dmp_orn.goal)
        err_abs_q = np.nan_to_num(err_abs_q)
        err_rel_q = err_abs_q / (np.linalg.norm(dmp_orn.goal - dmp_orn.y0) + 1e-14)
        err_rel_q = np.nan_to_num(err_rel_q)
        if t_q % 1000 == 0:
            print("t_q =", t_q, "err_abs_q =", err_abs_q, "err_rel_q =", err_rel_q)
        flag_q = ((t_q >= dmp_orn.cs.timesteps) or err_rel_q <= 0.01)
    print("t_q = ", t_q)

    e_track = []
    for q in q_track:
        e_x, e_y, e_z = rm.euler_from_quaternion(q)
        e_track.append(np.array([np.rad2deg(e_x), np.rad2deg(e_y), np.rad2deg(e_z)]))

    p_track = np.asarray(p_track)
    q_track = np.asarray(q_track)
    e_track = np.asarray(e_track)
    ###########################################
    fig = plt.figure(figsize=(8, 6))
    aa = fig.add_subplot(111, projection='3d', aspect="auto")  # change to 3d plot
    plt.plot(position[:,0], position[:,1], position[:,2], 'b', label='original trajectory')
    # plt.plot(y_des[0], y_des[1], 'orange', label='learned trajectory')

    plt.plot(p_track[:, 0], p_track[:, 1], p_track[:, 2], '--r', label='DMP++')
    plt.plot(dmp_pos.goal[0], dmp_pos.goal[1], dmp_pos.goal[2], '*g', markersize=10, label='goal')

    # plt.plot(dmp_new_under[:, 0], dmp_new_under[:, 1], dmp_new_under[:, 2], '--r', label='DMP++_under')
    # plt.plot(g_under[0], g_under[1], g_under[2], '*r', markersize=10, label='goal_under')
    plt.ylim(-10.0, 10.0)
    plt.setp(aa, xlabel='X', ylabel='Y', zlabel='Z')
    plt.legend(loc='best')
    # plt.axis('equal')
    plt.tight_layout()


    # plot
    x = np.linspace(0, 1, len(position))
    x1 = np.linspace(0, 1, len(p_track.T[0]))
    fig_p, axes_p = plt.subplots(nrows=3, ncols=2)
    fig_p.suptitle('Positions (mm)', fontweight='bold')
    axes_p[0, 0].plot(x, position.T[0], '--', color='r', label='p_x'); axes_p[0, 0].legend()
    axes_p[1, 0].plot(x, position.T[1], '--', color='g', label='p_y'); axes_p[1, 0].legend()
    # axes_p[1, 0].set_ylim(-3, 3)
    axes_p[2, 0].plot(x, position.T[2], '--', color='b', label='p_z'); axes_p[2, 0].legend()
    axes_p[0, 1].plot(x1, p_track.T[0], '-', color='r', label='p_new_x'); axes_p[0, 1].legend()
    axes_p[1, 1].plot(x1, p_track.T[1], '-', color='g', label='p_new_y'); axes_p[1, 1].legend()
    # axes_p[1, 1].set_ylim(-3, 3)
    axes_p[2, 1].plot(x1, p_track.T[2], '-', color='b', label='p_new_z'); axes_p[2, 1].legend()
    fig_p.tight_layout()

    x = np.linspace(0, 1, int(1 / dt))
    x1 = np.linspace(0, 1, len(q_track.T[0]))
    fig_q, axes_q = plt.subplots(nrows=4, ncols=2)
    fig_q.suptitle('Unit Quaternions', fontweight='bold')
    axes_q[0, 0].plot(x, q_des.T[0], '--', color='c', label='q_w'); axes_q[0, 0].legend()
    axes_q[1, 0].plot(x, q_des.T[1], '--', color='r', label='q_x'); axes_q[1, 0].legend()
    axes_q[2, 0].plot(x, q_des.T[2], '--', color='g', label='q_y'); axes_q[2, 0].legend()
    axes_q[3, 0].plot(x, q_des.T[3], '--', color='b', label='q_z'); axes_q[3, 0].legend()
    axes_q[0, 1].plot(x1, q_track.T[0], '-', color='c', label='q_new_w'); axes_q[0, 1].legend()
    axes_q[1, 1].plot(x1, q_track.T[1], '-', color='r', label='q_new_x'); axes_q[1, 1].legend()
    axes_q[2, 1].plot(x1, q_track.T[2], '-', color='g', label='q_new_y'); axes_q[2, 1].legend()
    axes_q[3, 1].plot(x1, q_track.T[3], '-', color='b', label='q_new_z'); axes_q[3, 1].legend()
    fig_q.tight_layout()

    x = np.linspace(0, 1, len(euler))
    x1 = np.linspace(0, 1, len(e_track.T[0]))
    fig_e, axes_e = plt.subplots(nrows=3, ncols=2)
    fig_e.suptitle('Euler angles ($^\circ$)', fontweight='bold')
    axes_e[0, 0].plot(x, euler.T[0], '--', color='r', label='eul_R'); axes_e[0, 0].legend()
    # axes_e[0, 0].set_ylim(-0.5, 0.5)
    axes_e[1, 0].plot(x, euler.T[1], '--', color='g', label='eul_P'); axes_e[1, 0].legend()
    axes_e[2, 0].plot(x, euler.T[2], '--', color='b', label='eul_Y'); axes_e[2, 0].legend()
    # axes_e[2, 0].set_ylim(-0.5, 0.5)
    axes_e[0, 1].plot(x1, e_track.T[0], '-', color='r', label='eul_new_R'); axes_e[0, 1].legend()
    axes_e[0, 1].set_xlim(0.0, 1.0)
    # axes_e[0, 1].set_ylim(-0.5, 0.5)
    axes_e[1, 1].plot(x1, e_track.T[1], '-', color='g', label='eul_new_P'); axes_e[1, 1].legend()
    axes_e[1, 1].set_xlim(0.0, 1.0)
    axes_e[2, 1].plot(x1, e_track.T[2], '-', color='b', label='eul_new_Y'); axes_e[2, 1].legend()
    # axes_e[2, 1].set_ylim(-0.5, 0.5)
    fig_e.tight_layout()
    plt.show()


