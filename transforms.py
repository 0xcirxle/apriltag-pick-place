# transforms.py
# small helpers for 4x4 homogeneous transforms so the other files stay readable

import numpy as np


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv(T):
    """inverse of a homogeneous transform without actually inverting the matrix:
       [R t]^-1 = [R' -R't]"""
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def pose_xy_yaw(x, y, z, yaw):
    """transform for something lying flat on the table at (x,y,z) rotated by yaw about z"""
    return make_T(rotz(yaw), [x, y, z])


def yaw_of(T):
    """heading of the x axis of T in the base xy plane"""
    return np.arctan2(T[1, 0], T[0, 0])


def apply(T, p):
    """apply T to a point or (N,3) points"""
    p = np.atleast_2d(p)
    ph = np.c_[p, np.ones(len(p))]
    out = (T @ ph.T).T[:, :3]
    return out[0] if out.shape[0] == 1 else out


def rot_angle(R):
    """angle of a rotation matrix (equivalent axis-angle), used for orientation error"""
    c = (np.trace(R) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))
