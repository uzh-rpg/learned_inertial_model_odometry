"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Some IMU-related functions

Reference frames:
W: fixed world frame
B: moving body (= imu) frame

"""

import numpy as np

from learning.utils.math import mat_exp


def propagate(init_state_dic, imu_dic, ba=None, bg=None):
    g = np.array([0, 0, -9.8082])

    ts = imu_dic["ts"]
    acc = imu_dic["accels"]
    gyr = imu_dic["gyros"]

    t0 = init_state_dic["ts"]
    p0 = init_state_dic["pos"]
    R0 = init_state_dic["ori"]
    v0 = init_state_dic["vel"]

    if ba == None:
        ba = np.zeros((3,))
    if bg == None:
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
        a0 = acc[i-1, :]
        
        w1 = gyr[i, :]
        a1 = acc[i, :]

        w = 0.5 * (w0 + w1) - bg
        a = 0.5 * (a0 + a1) - ba

        # propagate rot
        dtheta = w * dt
        dR = mat_exp(dtheta)
        R_wbi = R_wb[-1] @ dR
        
        # propagate vel and pos
        Rmid = 0.5 * (R_wb[-1] + R_wbi)
        dv = Rmid @ (a * dt)
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

