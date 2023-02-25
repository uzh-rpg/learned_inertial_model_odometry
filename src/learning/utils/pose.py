"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/uzh-rpg/uzh_fpv/blob/full-batch-optimization/python/uzh_fpv/pose.py
"""

import numpy as np
import pyquaternion
import scipy.linalg

import learning.utils.transformations as tf


def cross2Matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def matrix2Cross(M):
    skew = (M - M.T)/2
    return np.array([-skew[1, 2], skew[0, 2], -skew[0, 1]])


class Pose(object):
    def __init__(self, R, t):
        assert type(R) is np.ndarray
        assert type(t) is np.ndarray
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        self.R = R
        self.t = t

    def inverse(self):
        return Pose(self.R.T, -np.dot(self.R.T, self.t))

    def __mul__(self, other):
        if isinstance(other, Pose):
            return Pose(np.dot(self.R, other.R), np.dot(self.R, other.t) + self.t)
        if type(other) is np.ndarray:
            assert len(other.shape) == 2
            assert other.shape[0] == 3
            return np.dot(self.R, other) + self.t
        raise Exception('Multiplication with unknown type!')

    def asArray(self):
        return np.vstack((np.hstack((self.R, self.t)), np.array([0, 0, 0, 1])))
        
    def asTwist(self):
        so_matrix = scipy.linalg.logm(self.R)
        if np.sum(np.imag(so_matrix)) > 1e-10:
            raise Exception('logm called for a matrix with angle Pi. ' +
                'Not defined! Consider using another representation!')
        so_matrix = np.real(so_matrix)
        return np.hstack((np.ravel(self.t), matrix2Cross(so_matrix)))

    def q_wxyz(self):
        return pyquaternion.Quaternion(matrix=self.R).unit.q

    def fix(self):
        self.R = fixRotationMatrix(self.R)

    def fixed(self):
        return Pose(fixRotationMatrix(self.R), self.t)

    def __repr__(self):
        return self.asArray().__repr__()


def fromTwist(twist):
    # Using Rodrigues' formula
    w = twist[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return Pose(np.eye(3), twist[:3].reshape(3, 1))
    M = cross2Matrix(w/theta)
    R = np.eye(3) + M * np.sin(theta) + np.dot(M, M) * (1 - np.cos(theta))
    return Pose(R, twist[:3].reshape((3, 1)))


def fromPositionAndQuaternion(xyz, q_wxyz):
    R = pyquaternion.Quaternion(
        q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]).rotation_matrix
    t = xyz.reshape(3, 1)
    return Pose(R, t)


# ROS geometry_msgs/Pose
def fromPoseMessage(pose_msg):
    pos = pose_msg.position
    ori = pose_msg.orientation
    R = pyquaternion.Quaternion(ori.w, ori.x, ori.y, ori.z).rotation_matrix
    t = np.array([pos.x, pos.y, pos.z]).reshape(3, 1)
    return Pose(R, t)


def identity():
    return Pose(np.eye(3), np.zeros((3, 1)))


def geodesicDistanceSO3(R1, R2):
    return getRotationAngle(np.dot(R1, R2.T))


def getRotationAngle(R):
    return np.arccos((np.trace(R) - 1 - 1e-6) / 2)
    

def fixRotationMatrix(R):
    u, _, vt = np.linalg.svd(R)
    R_new = np.dot(u, vt)
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
    return R_new


def similarityTransformation(T_A_B, p_B_C, scale):
    return scale * np.dot(T_A_B.R, p_B_C) + T_A_B.t # 3x1 np array


def similarityTransformationPose(T_A_B, T_B_C, scale):
    R_A_C = np.dot(T_A_B.R, T_B_C.R)
    t_A_C = scale * np.dot(T_A_B.R, T_B_C.t) + T_A_B.t
    return Pose(R_A_C, t_A_C)


def loadFromTxt(fn):
    T = np.loadtxt(fn)
    return Pose(T[0:3, 0:3], np.array(T[0:3, 3]).reshape(3,1))


def fromAngleAxisToRotMat(theta, axis):
    axis = axis / np.linalg.norm(axis)
    R = np.cos(theta) * np.eye(3) - np.sin(theta) * cross2Matrix(axis) + \
        (1 - np.cos(theta)) * np.outer(axis,axis)
    return R 


# [in] = np.array([[qx, qy, qz, qw], [], ...])
# [out] = np.array([[rotz, roty, rotx], [], ...])
def fromQuatToEulerAng(q_xyzw):
    R = pyquaternion.Quaternion(\
        np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])).rotation_matrix
    rz, ry, rx = tf.euler_from_matrix(R, 'rzyx')
    rot_zyx = np.array([rz, ry, rx])
    rot_zyx = np.rad2deg(rot_zyx)
    return rot_zyx


def xyzwQuatTowxyzQuat(q_xyzw):
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def xyzwQuatToMat(q_xyzw):
    q_wxyz = xyzwQuatTowxyzQuat(q_xyzw)
    return pyquaternion.Quaternion(q_wxyz).rotation_matrix


def xyzwQuatFromMat(R):
    # check R as in pyquaternion/quaternion.py (line 180).
    if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), rtol=1e-5, atol=1e-6):
        R = fixRotationMatrix(R)
    q_wxyz = pyquaternion.Quaternion(matrix=R, atol=1e-6)
    q_xyzw = \
        np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return q_xyzw

