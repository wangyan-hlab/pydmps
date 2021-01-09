import numpy as np
import pydmps.dmp_discrete_modified as dmp_p
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set('talk', 'whitegrid', 'bright', font_scale=0.7,
        rc={"lines.linewidth": 1.5, 'grid.linestyle': '--'})

timesteps = 1000
dt = 0.01
ax = 10.0
n_bfs = 50
n_dmps = 2
k = 1000
alpha = 25*np.ones(n_dmps)

t = np.linspace(0.0, np.pi, timesteps)
x = t
y = np.sin(t) * np.sin(t) + t / 15.0 / np.pi
# y = (np.sin(t)) ** 2.0 + t / 15.0 / np.pi
gamma = np.transpose(np.array([x, y]))

y0_old = gamma[0]
g_old = gamma[-1]
# print('g_old =', g_old)

dmp_new = dmp_p.DMPs_discrete_modified(dt=dt, n_dmps=n_dmps, n_bfs=n_bfs, ax=ax, K=k,
                                       form='mod', rescale='rotodilatation')
dmp_new_norescale = dmp_p.DMPs_discrete_modified(dt=dt, n_dmps=n_dmps, n_bfs=n_bfs, ax=ax, K=k,
                                                 form='mod', rescale=None)
dmp_old = dmp_p.DMPs_discrete_modified(dt=dt, n_dmps=n_dmps, n_bfs=n_bfs, ax=ax, K=k, form='old')
y_des_new = dmp_new.imitate_path(y_des=gamma)
y_des_old = dmp_old.imitate_path(y_des=gamma)
y_des_new_norescale = dmp_new_norescale.imitate_path(y_des=gamma)

# set higher goal
theta = 140 * np.pi / 180.0
R_h = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
g_high = np.dot(R_h, g_old)
# set lower goal
theta = -45 * np.pi / 180.0
R_u = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
g_under = np.dot(R_u, g_old)

# change start point
dmp_new.y0 = y0_old + np.array([-4,0])
dmp_new_norescale.y0 = y0_old + np.array([-4,0])
dmp_old.y0 = y0_old + np.array([-4,0])
# change goal point
dmp_new.goal = g_high
dmp_new_norescale.goal = g_high
dmp_old.goal = g_high

dmp_new_high, _, _ = dmp_new.rollout()
dmp_new_norescale_high, _, _ = dmp_new_norescale.rollout()
dmp_old_high, _, _ = dmp_old.rollout()

dmp_new.goal = g_under
dmp_new_norescale.goal = g_under
dmp_old.goal = g_under
dmp_new_under, _, _ = dmp_new.rollout()
dmp_new_norescale_under, _, _ = dmp_new_norescale.rollout()
dmp_old_under, _, _ = dmp_old.rollout()

plt.figure(1, figsize=(6, 6))
plt.plot(x, y, 'b', label='original trajectory')
plt.plot(g_old[0], g_old[1], '*b', markersize=10, label='goal_old')
# plt.plot(y_des[0], y_des[1], 'orange', label='learned trajectory')
plt.plot(dmp_new_high[:,0], dmp_new_high[:,1], '--g', label='high, rotodila')    # Ginesi 2019
plt.plot(dmp_new_norescale_high[:,0], dmp_new_norescale_high[:,1], ':g', label='high, w/o rotodila')    # Park 2008
# plt.plot(dmp_old_high[:,0], dmp_old_high[:,1], ':k', label='Ijspeert_high')    # Ijspeert 2013
plt.plot(g_high[0], g_high[1], '*g', markersize=10, label='goal_high')
plt.plot(dmp_new_under[:,0], dmp_new_under[:,1], '--r', label='under, rotodila')    # Ginesi 2019
plt.plot(dmp_new_norescale_under[:,0], dmp_new_norescale_under[:,1], ':r', label='under, w/o rotodila')    # Park 2008
# plt.plot(dmp_old_under[:,0], dmp_old_under[:,1], ':c', label='Ijspeert_under')    # Ijspeert 2013
plt.plot(g_under[0], g_under[1], '*r', markersize=10, label='goal_under')
plt.legend(loc='best')
plt.axis('equal')
plt.tight_layout()

plt.figure(2, figsize=(6, 6))
color = ['aqua','brown','chartreuse','coral','darkblue','lavender','lime','maroon','olive','sienna','m','r','g']
cind = color.index('r')
cind_1 = color.index('g')
for tt in range(0, 360, 10):
        theta = tt * np.pi / 180.0
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        g_new = np.dot(R, g_old)
        # plot trajectories from DMPs with rotodilatation
        dmp_new.goal = g_new
        dmp_new_goal, _, _ = dmp_new.rollout()
        plt.plot(dmp_new_goal[:, 0], dmp_new_goal[:, 1], '-', color=color[cind], lw=2, label='DMP++, rotodila')

        plt.plot(g_new[0], g_new[1], '*', color=color[cind], markersize=10)
        # plot trajectories from DMPs without rotodilatation
        dmp_new_norescale.goal = g_new
        dmp_new_norescale_goal, _, _ = dmp_new_norescale.rollout()
        plt.plot(dmp_new_norescale_goal[:, 0], dmp_new_norescale_goal[:, 1], '--', color=color[cind_1], lw=1.5, label='DMP++, w/o rotodila')
        plt.plot(g_new[0], g_new[1], '*', color=color[cind_1], markersize=10)

handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.axis('equal')
plt.tight_layout()
plt.show()