"""
Copyright (C) 2013 Travis DeWolf

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
from pydmps.cs import CanonicalSystem
from pydmps.rotation_matrix import roto_dilatation
import utiltools.robotmath as rm
import pydmps.quaternion_operation as q_oper
from pyquaternion import Quaternion
import scipy.integrate
import numpy as np
import copy

class DMPs_discrete_modified(object):
    """
        An implementation of Cartesian discrete DMPs
        Modified based on DMP++:
        https://github.com/mginesi/dmp_pp

        Using quaternion (w,i,j,k) DMPs for orientation in Cartesian space.
    """

    def __init__(self, n_dmps, n_bfs, dt=0.01, ax=4.0, y0=0, goal=1, w=None, K=1050, D=None,
                 form='mod', basis='gaussian', rescale='rotodilatation', dim='position', **kwargs):
        """
        """

        self.n_dmps = copy.deepcopy(n_dmps)
        self.n_bfs = n_bfs
        self.dt = copy.deepcopy(dt)
        self.ax = copy.deepcopy(ax)
        self.form = copy.deepcopy(form)
        self.rescale = copy.deepcopy(rescale)
        self.basis = copy.deepcopy(basis)
        self.dim = copy.deepcopy(dim)
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        if w is None:
            w = np.zeros((self.n_dmps, self.n_bfs)) if self.dim == 'position' \
            else np.zeros((self.n_dmps-1, self.n_bfs))
        self.w = copy.deepcopy(w)
        self.K = copy.deepcopy(K)
        if D is None:
            D = 2 * np.sqrt(self.K)
        self.D = copy.deepcopy(D)
        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, ax=self.ax, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        self.gen_centers()
        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.gen_width()
        # set up the DMP system
        self.reset_state()

        # self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        # self.check_offset()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4

    def rollout(self, timesteps=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        self.reset_state()

        if timesteps is None:
            if "tau" in kwargs:
                timesteps = int(self.timesteps / kwargs["tau"])
            else:
                timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):

            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def gen_width(self):
        '''
        Set the "widths" for the basis functions.
        '''
        if self.basis == 'gaussian':
            self.h = 1.0 / np.diff(self.c) / np.diff(self.c)
            self.h = np.append(self.h, self.h[-1])
        else:
            self.h = 0.2/ np.diff(self.c)
            self.h = np.append(self.h[0], self.h)

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        self.c = np.exp(- self.cs.ax * self.cs.run_time *
                        ((np.cumsum(np.ones([1, self.n_bfs])) - 1) / self.n_bfs))

    def gen_front_term(self, x, n):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        if self.dim == 'position':
            return x * (self.goal - self.y0)
        if self.dim == 'orientation':
            return x * q_oper.quaternion_error(self.goal, self.y0)

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[-1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """
        c = np.reshape(self.c, [self.n_bfs, 1])
        h = np.reshape(self.h, [self.n_bfs, 1])
        if self.basis == 'gaussian':
            xi = h * (x - c)**2
            psi_set = np.exp(-xi)
        else:
            xi = np.abs(h * (x - c))
            if self.basis == 'mollifier':
                psi_set = (np.exp(- 1.0 / (1.0 - xi * xi))) * (xi < 1.0)
            elif self.basis == 'wendland2':
                psi_set = ((1.0 - xi) ** 2.0) * (xi < 1.0)
            elif self.basis == 'wendland3':
                psi_set = ((1.0 - xi) ** 3.0) * (xi < 1.0)
            elif self.basis == 'wendland4':
                psi_set = ((1.0 - xi) ** 4.0 * (4.0 * xi + 1.0)) * (xi < 1.0)
            elif self.basis == 'wendland5':
                psi_set = ((1.0 - xi) ** 5.0 * (5.0 * xi + 1)) * (xi < 1.0)
            elif self.basis == 'wendland6':
                psi_set = ((1.0 - xi) ** 6.0 *
                    (35.0 * xi ** 2.0 + 18.0 * xi + 3.0)) * (xi < 1.0)
            elif self.basis == 'wendland7':
                psi_set = ((1.0 - xi) ** 7.0 *
                    (16.0 * xi ** 2.0 + 7.0 * xi + 1.0)) * (xi < 1.0)
            elif self.basis == 'wendland8':
                psi_set = (((1.0 - xi) ** 8.0 *
                    (32.0 * xi ** 3.0 + 25.0 * xi ** 2.0 + 8.0 * xi + 1.0)) * (xi < 1.0))
        psi_set = np.nan_to_num(psi_set)
        return psi_set

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        # f_target = np.transpose(f_target)
        s_track = self.cs.rollout()
        psi_track = self.gen_psi(s_track)
        # Compute useful quantities
        sum_psi = np.sum(psi_track, 0)
        sum_psi_2 = sum_psi * sum_psi
        s_track_2 = s_track * s_track
        # Set up the minimization problem
        A = np.zeros([self.n_bfs, self.n_bfs])
        b = np.zeros([self.n_bfs])
        # The matrix does not depend on f
        for k in range(self.n_bfs):
            A[k, k] = scipy.integrate.simps(
                psi_track[k] * psi_track[k] * s_track_2 / sum_psi_2, s_track)
            for h in range(k + 1, self.n_bfs):
                A[k, h] = scipy.integrate.simps(
                    psi_track[k] * psi_track[h] * s_track_2 / sum_psi_2, s_track)
                A[h, k] = A[k, h].copy()
        LU = scipy.linalg.lu_factor(A)
        # The problem is decoupled for each dimension
        if self.dim == 'position':
            for d in range(self.n_dmps):
                # Create the vector of the regression problem
                for k in range(self.n_bfs):
                    b[k] = scipy.integrate.simps(
                        f_target[d] * psi_track[k] * s_track / sum_psi, s_track)
                # Solve the minimization problem
                self.w[d] = scipy.linalg.lu_solve(LU, b)
        else:
            for d in range(self.n_dmps - 1):
                # Create the vector of the regression problem
                for k in range(self.n_bfs):
                    b[k] = scipy.integrate.simps(
                        f_target[d] * psi_track[k] * s_track / sum_psi, s_track)
                # Solve the minimization problem
                self.w[d] = scipy.linalg.lu_solve(LU, b)
        self.w = np.nan_to_num(self.w)

    def imitate_path(self, y_des, dy_des=None, ddy_des=None, t_des=None, plot=False):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [run_time, n_dmps]
        """

        # set initial state and goal
        self.y0 = y_des[0].copy()
        self.goal = y_des[-1].copy()

        # self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        if t_des is None:
            # Default value for t_des
            t_des = np.linspace(0, self.cs.run_time, y_des.shape[0])
        else:
            # Warp time to start from zero and end up to T
            t_des -= t_des[0]
            t_des /= t_des[-1]
            t_des *= self.cs.run_time
        time = np.linspace(0, self.cs.run_time, self.cs.timesteps)

        path_gen = scipy.interpolate.interp1d(t_des, y_des.transpose())
        path = path_gen(time)
        y_des = path.transpose()    # timesteps x n_dmps

        # calculate velocity of y_des with central differences
        if self.dim == 'position':
            if dy_des is None:
                dy_des = np.zeros([self.n_dmps, self.cs.timesteps])
                for i in range(self.n_dmps):
                    dy_des[i] = np.gradient(path[i])/self.dt
                    # dy_des = np.gradient(y_des, axis=1) / self.dt
            # calculate acceleration of y_des with central differences
            if ddy_des is None:
                ddy_des = np.zeros([self.n_dmps, self.cs.timesteps])
                for i in range(self.n_dmps):
                    ddy_des[i] = np.gradient(dy_des[i]) / self.dt
            dy_des = dy_des.transpose()     # timesteps x n_dmps
            ddy_des = ddy_des.transpose()

            # find the force required to move along this trajectory
            f_target = np.zeros([self.n_dmps, self.cs.timesteps])
            if self.form == 'mod':
                f_target = ((ddy_des/self.K - (self.goal - y_des) + self.D * dy_des/self.K).transpose() +
                             np.reshape((self.goal - self.y0), [self.n_dmps, 1]) * self.cs.rollout())   # n_dmps x timesteps
            elif self.form == 'old':
                # f_target = ((ddy_des - self.K * (self.goal - y_des) + self.D * dy_des)/(self.goal - self.y0)).transpose()
                f_target = np.dot(np.linalg.inv(np.diag(self.goal - self.y0)),
                                  (ddy_des - self.K * (self.goal - y_des) + self.D * dy_des).transpose())

        elif self.dim == 'orientation':
            dq_des = np.zeros([self.n_dmps, self.cs.timesteps])
            # # Ude, 2014, "Orientation in Cartesian Space Dynamic Movement Primitives"
            angular_vel = np.zeros([self.timesteps, self.n_dmps-1])
            d_angular_vel = np.zeros([self.n_dmps-1, self.timesteps])

            for i in range(self.n_dmps):
                dq_des[i] = np.gradient(path[i])/self.dt    # dq, n_dmps x timesteps

            for index, dq in enumerate(dq_des.T):
                q = path.T[index]
                angular_vel[index] = 2 * rm.quaternion_multiply(dq, rm.quaternion_conjugate(q))[1:]  # w(t)=Im(2*dq(t)*q_conj(t))

            for i in range(self.n_dmps - 1):
                d_angular_vel[i] = np.gradient(angular_vel.T[i])/self.dt
            if dy_des is None:
                dy_des = angular_vel
            if ddy_des is None:
                ddy_des = d_angular_vel
            ddy_des = ddy_des.transpose()   # d_eta, timesteps x n_dmps

            import matplotlib.pyplot as plt

            # fig, axes = plt.subplots(3, 2)
            # axes[0, 0].plot(dy_des[:, 0], '--r', label='$\eta_x$'); axes[0, 0].legend()
            # axes[1, 0].plot(dy_des[:, 1], '--g', label='$\eta_y$'); axes[1, 0].legend()
            # axes[2, 0].plot(dy_des[:, 2], '--b', label='$\eta_z$'); axes[2, 0].legend()
            # axes[0, 1].plot(ddy_des[:, 0], '-r', label='$\dot{\eta}_x$'); axes[0, 1].legend()
            # axes[1, 1].plot(ddy_des[:, 1], '-g', label='$\dot{\eta}_y$'); axes[1, 1].legend()
            # axes[2, 1].plot(ddy_des[:, 2], '-b', label='$\dot{\eta}_z$'); axes[2, 1].legend()
            # fig.tight_layout()

            f_target = np.zeros([self.cs.timesteps, self.n_dmps-1])
            eq_g_0 = q_oper.quaternion_error(self.goal, self.y0)
            for t in range(self.cs.timesteps):
                eq = q_oper.quaternion_error(self.goal, y_des[t])
                f_target[t] = np.dot(np.linalg.inv(np.diag(eq_g_0)), ddy_des[t] - self.K * eq + self.D * dy_des[t])

            # # Koutras, 2019, "A correct formulation for the orientation DMPs for robot control in the Cartesian space"
            # if dy_des is None:
            #     dy_des = np.zeros([self.n_dmps-1, self.cs.timesteps])
            #
            #     for index, dq in enumerate(dq_des.T):
            #         q = path.T[index]
            #         q_conj = rm.quaternion_conjugate(q)
            #         j_q = q_oper.calculate_jacobian(rm.quaternion_multiply(self.goal, q_conj))[1]  # Jacobian_q matrix [3x4]
            #         d_eq = -2*np.dot(j_q, rm.quaternion_multiply(self.goal,
            #                          rm.quaternion_multiply(q_conj, rm.quaternion_multiply(dq, q_conj))).transpose())
            #         dy_des[:, index] = d_eq.flatten()
            # if ddy_des is None:
            #     ddy_des = np.zeros([self.n_dmps-1, self.cs.timesteps])
            #     for i in range(self.n_dmps-1):
            #         ddy_des[i] = np.gradient(dy_des[i]) / self.dt
            # dy_des = dy_des.transpose()  # timesteps x n_dmps
            # ddy_des = ddy_des.transpose()
            # f_target = np.zeros([self.n_dmps-1, self.cs.timesteps])
            # eq_g_0 = q_oper.quaternion_error(self.goal, self.y0)
            # f_target = f_target.T
            # for t in range(self.cs.timesteps):
            #     eq = q_oper.quaternion_error(self.goal, y_des[t])
            #
            #     f_target[t] = np.dot(np.diag(eq_g_0), ddy_des[t] - self.K*eq + self.D*dy_des[t])

            f_target = f_target.T

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        if plot is True:
            # plot the basis function activations
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(211)
            psi_track = self.gen_psi(self.cs.rollout())
            plt.plot(np.transpose(psi_track))
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[ii,:], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(
                    np.sum(np.transpose(psi_track) * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset_state()
        self.learned_position = self.goal - self.y0

        return y_des

    def rollout(self, timesteps=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        self.reset_state()

        if timesteps is None:
            if "tau" in kwargs:
                timesteps = int(self.timesteps / kwargs["tau"])
            else:
                timesteps = self.timesteps

        # set up tracking vectors
        if self.dim == 'orientation':
            y_track = np.zeros((timesteps, self.n_dmps))    # quaternion
            dy_track = np.zeros((timesteps, self.n_dmps - 1))
            ddy_track = np.zeros((timesteps, self.n_dmps - 1))
        elif self.dim == 'position':
            y_track = np.zeros((timesteps, self.n_dmps))
            dy_track = np.zeros((timesteps, self.n_dmps))
            ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):

            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        if self.dim == 'position':
            self.dy = np.zeros(self.n_dmps)
            self.ddy = np.zeros(self.n_dmps)
        elif self.dim == 'orientation':
            self.dy = np.zeros(self.n_dmps - 1)
            self.ddy = np.zeros(self.n_dmps - 1)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """
        if self.rescale == 'rotodilatation':
            M = roto_dilatation(self.learned_position, self.goal - self.y0)
        elif self.rescale == 'diagonal':
            M = np.diag((self.goal - self.y0) / self.learned_position)
        else:
            M = np.eye(self.n_dmps)

        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)
        if self.dim == 'position':
            if self.form == 'mod':
                # generate the forcing term
                f = (np.dot(self.w, psi[:, 0])) / np.sum(psi) * x
                f = np.nan_to_num(np.dot(M, f))
                # DMP acceleration
                self.ddy = (self.K*(self.goal-self.y) - self.D*self.dy -
                            self.K*(self.goal-self.y0)*x + self.K*f)
                if external_force is not None:
                    self.ddy += external_force
                self.dy += self.ddy * tau * self.dt * error_coupling
                self.y += self.dy * tau * self.dt * error_coupling
            else:
                # generate the forcing term
                f = self.gen_front_term(x, self.n_dmps) * (np.dot(self.w, psi[:,0])) / np.sum(psi)
                # DMP acceleration
                self.ddy = (self.K * (self.goal - self.y) - self.D * self.dy + f)
                if external_force is not None:
                    self.ddy += external_force
                self.dy += self.ddy * tau * self.dt * error_coupling
                self.y += self.dy * tau * self.dt * error_coupling

        elif self.dim == 'orientation':
            # generate the forcing term
            f = self.gen_front_term(x, self.n_dmps - 1) * (np.dot(self.w, psi[:, 0])) / np.sum(psi)

            # # Ude, 2014, "Orientation in Cartesian Space Dynamic Movement Primitives"
            self.ddy = (self.K * q_oper.quaternion_error(self.goal, self.y) - self.D * self.dy + f)
            if external_force is not None:
                self.ddy += external_force
            self.dy += self.ddy * tau * self.dt * error_coupling
            # w_q = np.concatenate((np.array([0]), self.dy), axis=0)  # w_q = [0, w]
            # dq = (rm.quaternion_multiply(1 / 2 * w_q, self.y) * tau * error_coupling).flatten()

            q_delta = (q_oper.vector_exp(1 / 2 * self.dy * tau * error_coupling * self.dt)).flatten()
            self.y = rm.quaternion_multiply(q_delta, self.y)

            # Koutras, 2019, "A correct formulation for the orientation DMPs for robot control in the Cartesian space"
            # # DMP acceleration
            # eq = q_oper.quaternion_error(self.goal, self.y)
            # self.ddy = (self.K * (0 - eq) - self.D * self.dy + f)
            # # print('ddy =', self.ddy)
            # if external_force is not None:
            #     self.ddy += external_force
            # self.dy += self.ddy * tau * self.dt * error_coupling
            # # print('dy =', self.dy)
            # q_conj = rm.quaternion_conjugate(self.y)
            # goal_conj = rm.quaternion_conjugate(self.goal)
            # dq = -1/2 * self.y * goal_conj * \
            #     np.dot(q_oper.calculate_jacobian(rm.quaternion_multiply(self.goal, q_conj))[0], self.dy) * self.y
            # self.y += dq * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy
# ==============================
# Test code
# ==============================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    sns.set()
    sns.set('talk', 'whitegrid', 'bright', font_scale=0.7,
            rc={"lines.linewidth": 1.5, 'grid.linestyle': '--'})

    timesteps = 1000
    dt = 0.001
    ax = 1.0
    n_bfs = 100
    # n_dmps = 3
    k = 1000

    t = np.linspace(0.0, np.pi, timesteps)
    x = t
    y = np.sin(t) * np.sin(t) + t / 15.0 / np.pi
    z = t**3
    gamma = np.transpose(np.array([x, y, z]))
    g_old = gamma[-1]

    dmp_new = DMPs_discrete_modified(dt=dt, n_bfs=n_bfs, ax=ax, K=k, form='mod', dim='position',
                                     rescale='rotodilatation', n_dmps=3)
    # dmp_old = DMPs_discrete_modified(dt=dt, n_bfs=n_bfs, ax=ax, K=k, form='mod', dim='position',
    #                                  rescale=None, n_dmps=3)
    dmp_old = DMPs_discrete_modified(dt=dt, n_bfs=n_bfs, ax=ax, K=k, form='old', dim='position', n_dmps=3)
    y_des_new = dmp_new.imitate_path(y_des=gamma)
    y_des_old = dmp_old.imitate_path(y_des=gamma)

    g_high = g_old + np.array([0, g_old[1], 0])
    g_under = g_old * np.array([1, -10, 1])
    dmp_new.goal = g_high
    dmp_old.goal = g_high
    dmp_new_high, _, _ = dmp_new.rollout()
    dmp_old_high, _, _ = dmp_old.rollout()

    dmp_new.goal = g_under
    dmp_old.goal = g_under
    dmp_new_under, _, _ = dmp_new.rollout()
    dmp_old_under, _, _ = dmp_old.rollout()

    fig = plt.figure(figsize=(8, 6))
    aa = fig.add_subplot(111, projection='3d', aspect="auto")    # change to 3d plot
    plt.plot(x, y, z, 'b', label='original trajectory')
    # plt.plot(y_des[0], y_des[1], 'orange', label='learned trajectory')

    plt.plot(dmp_new_high[:,0], dmp_new_high[:,1], dmp_new_high[:,2], '--g', label='DMP++_high')
    # print('DMP++_high', dmp_new_high[-1, 0], dmp_new_high[-1, 1])
    plt.plot(dmp_old_high[:,0], dmp_old_high[:,1], dmp_old_high[:,2], ':k', label='Ijspeert_high')
    # print('Ijs_high', dmp_old_high[-1, 0], dmp_old_high[-1, 1])
    plt.plot(g_high[0], g_high[1], g_high[2], '*g', markersize=10, label='goal_high')
    # print('goal_high', g_high[0], g_high[1])

    plt.plot(dmp_new_under[:,0], dmp_new_under[:,1], dmp_new_under[:,2], '--r', label='DMP++_under')
    # print('DMP++_under', dmp_new_under[-1, 0], dmp_new_under[-1, 1])
    plt.plot(dmp_old_under[:,0], dmp_old_under[:,1], dmp_old_under[:,2], ':c', label='Ijspeert_under')
    # print('Ijs_under', dmp_old_under[-1, 0], dmp_old_under[-1, 1])
    plt.plot(g_under[0], g_under[1], g_under[2], '*r', markersize=10, label='goal_under')
    # print('g_under', g_under[0], g_under[1])

    plt.setp(aa, xlabel='X', ylabel='Y', zlabel='Z')
    plt.legend(loc='best')
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()
