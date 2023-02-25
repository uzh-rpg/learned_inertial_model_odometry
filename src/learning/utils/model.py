"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Some quad-model-related functions

Reference frames:
W: fixed world frame
B: moving body frame

"""

import numpy as np

from scipy.spatial.transform import Rotation

from learning.utils.math import mat_exp


def propagate(init_state_dic, meas_dic, bias_gyro=None):
    g = np.array([0, 0, -9.8082])

    ts = meas_dic["ts"]
    thrusts = meas_dic["thrusts"]
    gyr = meas_dic["gyros"]

    t0 = init_state_dic["ts"]
    p0 = init_state_dic["pos"]
    R0 = init_state_dic["ori"]
    v0 = init_state_dic["vel"]

    if bias_gyro == None:
        bg = np.zeros((3,))        

    ts_b, p_wb, R_wb, v_wb = [], [], [], []
    # init. values
    ts_b.append(t0)
    p_wb.append(p0)
    R_wb.append(R0)
    v_wb.append(v0)

    t_prev = t0
    for i in range(1, len(ts)):
        dt = ts[i] - t_prev
        
        w0 = gyr[i-1, :]
        thr0 = thrusts[i-1]
        
        w1 = gyr[i, :]
        thr1 = thrusts[i]

        w = 0.5 * (w0 + w1) - bg
        thr = 0.5 * (thr0 + thr1)
        thr_vec = np.array([0., 0., thr])

        # propagate rot
        dtheta = w * dt
        dR = mat_exp(dtheta)
        R_wbi = R_wb[-1] @ dR
        
        # propagate vel and pos
        Rmid = 0.5 * (R_wb[-1] + R_wbi)
        dv = Rmid @ (thr_vec * dt)
        dp = 0.5 * dv * dt
        gdt = g * dt
        gdt2 = gdt * dt
        v_wbi = v_wb[-1] + dv + gdt
        p_wbi = p_wb[-1] + v_wb[-1] * dt + dp + 0.5 * gdt2

        ts_b.append(ts[i])
        p_wb.append(p_wbi)
        R_wb.append(R_wbi)
        v_wb.append(v_wbi)

        t_prev = ts[i]

    traj = {}
    traj["ts"] = ts_b
    traj["pos"] = p_wb
    traj["ori"] = R_wb
    traj["vel"] = v_wb
    
    return traj


def propagate_with_gt_orientations(init_state_dic, meas_dic):
    g = np.array([0, 0, -9.8082])

    ts = meas_dic["ts"]
    thrusts = meas_dic["thrusts"]
    qs = meas_dic["orientations"]

    t0 = init_state_dic["ts"]
    p0 = init_state_dic["pos"]
    R0 = init_state_dic["ori"]
    v0 = init_state_dic["vel"] 

    ts_b, p_wb, R_wb, v_wb = [], [], [], []
    # init. values
    ts_b.append(t0)
    p_wb.append(p0)
    R_wb.append(R0)
    v_wb.append(v0)

    t_prev = t0
    for i in range(1, len(ts)):
        dt = ts[i] - t_prev
        
        R0 = Rotation.from_quat(qs[i-1]).as_matrix()
        thr0 = thrusts[i-1]
        
        R1 = Rotation.from_quat(qs[i]).as_matrix()
        thr1 = thrusts[i]

        R = 0.5 * (R0 + R1)
        thr = 0.5 * (thr0 + thr1)
        thr_vec = np.array([0., 0., thr])
        
        # propagate vel and pos
        dv = R @ (thr_vec * dt)
        dp = 0.5 * dv * dt
        gdt = g * dt
        gdt2 = gdt * dt
        v_wbi = v_wb[-1] + dv + gdt
        p_wbi = p_wb[-1] + v_wb[-1] * dt + dp + 0.5 * gdt2

        ts_b.append(ts[i])
        p_wb.append(p_wbi)
        R_wb.append(R1)
        v_wb.append(v_wbi)

        t_prev = ts[i]

    traj = {}
    traj["ts"] = ts_b
    traj["pos"] = p_wb
    traj["ori"] = R_wb
    traj["vel"] = v_wb
    
    return traj

