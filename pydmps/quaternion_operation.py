import numpy as np
import utiltools.robotmath as rm

def vector_exp(vector):
    v_norm = np.linalg.norm(vector)
    quaternion = np.array([[1],[0],[0],[0]]) if np.linalg.norm(vector) < 1e-6 \
        else np.concatenate((np.array([np.cos(v_norm)]), np.sin(v_norm)*vector/v_norm), axis=0)

    return quaternion

def quaternion_log(quaternion):
    """ calculate the logarithm of a unit quaternion """

    real = rm.quaternion_real(quaternion)
    imag = rm.quaternion_imag(quaternion)
    log_q = np.zeros([3, 1]) if abs(imag.sum()) < 1e-5 else np.arccos(real)*imag/np.linalg.norm(imag)

    return log_q

def calculate_jacobian(quaternion):
    """ calculate the quaternion exponential """
    real = rm.quaternion_real(quaternion)
    imag = rm.quaternion_imag(quaternion)
    theta = np.arccos(real)
    n = imag/np.sin(theta)
    j_logq = np.concatenate((np.zeros([1, 3]), np.eye(3)), axis=0) if abs(theta) < 1e-5 \
        else np.concatenate(([np.reshape(-imag.T, [1, 3]), np.sin(theta)/theta*(np.eye(3)-n*n.T)+real*n*n.T]), axis=0)
    j_q = np.concatenate((np.zeros([3, 1]), np.eye(3)), axis=1) if abs(theta) < 1e-5 \
        else np.concatenate((np.reshape((-np.sin(theta)+theta*real)/np.square(np.sin(theta))*n, [3, 1]),
                            theta/np.sin(theta)*np.eye(3)), axis=1)

    return j_logq, j_q

def quaternion_error(q1, q0):

    product = rm.quaternion_multiply(q1, rm.quaternion_conjugate(q0))
    error_q = 2*quaternion_log(product)
    error_q = error_q.flatten()

    return error_q


if __name__ == '__main__':

    # print(calculate_jacobian(q))
    q1 = rm.random_quaternion()
    print(q1)
    q2 = q1 + [0, 0, 0, 0]
    print(q2)
    print(rm.quaternion_multiply(q1, rm.quaternion_conjugate(q2)))
    d = quaternion_error(q1, q2)
    print(d)
