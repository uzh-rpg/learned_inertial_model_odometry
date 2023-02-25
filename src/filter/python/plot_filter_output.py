"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import src.utils.plotting as plotting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    result_dir = args.result_dir
    out_dir = os.path.join(result_dir, args.dataset, args.seq, 'pyfilter')

    # load data
    traj_fn = os.path.join(out_dir, "stamped_traj_estimate.txt")
    traj = np.loadtxt(traj_fn)

    bias_fn = os.path.join(out_dir, "stamped_bias_estimate.txt")
    bias = np.loadtxt(bias_fn)
    ts = bias[:,0]
    bg = bias[:,1:4]
    ba = bias[:,4:]

    vel_fn = os.path.join(out_dir, "stamped_vel_estimate.txt")
    vel = np.loadtxt(vel_fn)

    gt_fn = os.path.join(dataset_dir, args.dataset, args.seq, 'stamped_groundtruth_imu.txt')
    gt_traj = np.loadtxt(gt_fn)
    gt_ts = gt_traj[:, 0]

    # sample at same times
    if traj[0,0] < gt_ts[0]:
        idxs = np.argwhere(traj > gt_ts[0])[0][0]
    else:
        idxs = 0
    if traj[-1,0] > gt_ts[-1]:
        idxe = np.argwhere(traj > gt_ts[-1])[0][0]
    else:
        idxe = traj.shape[0]
    traj = traj[idxs:idxe]

    gt_pos_data = interp1d(gt_traj[:,0], gt_traj[:,1:4], axis=0)(traj[:,0])
    gt_rot_data = Slerp(gt_traj[:,0], Rotation.from_quat(gt_traj[:,4:8]))(traj[:,0])
    gt_ts = traj[:,0]
    gt_traj = np.concatenate((
        gt_ts.reshape((-1,1)), gt_pos_data, gt_rot_data.as_quat()), axis=1)

    # get estimate of velocity
    dts = (gt_ts[2:] - gt_ts[:-2])
    vel_gt = (gt_traj[2:, 1:4] - gt_traj[:-2, 1:4]) / dts.reshape((-1,1))
    vel_gt = np.concatenate((gt_ts[1:-1].reshape((-1,1)), vel_gt), axis=1)

    # make plots
    plotting.make_position_plots(traj, gt_traj)
    plotting.plotBiases(ts, bg, ba)
    plotting.make_velocity_plots(vel, vel_gt)

    plt.show()

