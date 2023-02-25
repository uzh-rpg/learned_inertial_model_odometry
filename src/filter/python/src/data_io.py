"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Class to load input data
Reference: https://github.com/CathIAS/TLIO/blob/master/src/dataloader/data_io.py
"""

import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


class DataIO:
    def __init__(self, quad_name=None):
        self.quad_name = quad_name
        self.ts = None
        self.accel_raw = None
        self.gyro_raw = None
        self.accel_calib = None
        self.gyro_calib = None
        self.dataset_size = None
        self.gyro_bias = None
        self.accel_bias = None
        self.gt_traj = None

    def load(self, root_dir, dataset, sequence):
        """
        load data
        """
        indir = os.path.join(root_dir, dataset, sequence)
        with h5py.File(os.path.join(indir, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            accel_raw = np.copy(f["accel_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            accel_calib = np.copy(f["accel_calib"])
            i_thrust = np.copy(f["i_thrust"])
            gt_traj = np.copy(f["traj_target"])
            gyro_bias = np.copy(f["gyro_bias"])
            accel_bias = np.copy(f["accel_bias"])

        self.ts = np.round(ts, 5)
        self.accel_raw = accel_raw
        self.gyro_raw = gyro_raw
        self.accel_calib = accel_calib
        self.gyro_calib = gyro_calib
        self.thrust = i_thrust
        self.dataset_size = self.ts.shape[0]
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias
        self.gt_ts = np.round(ts, 5)
        self.gt_p = gt_traj[:, 0:3]
        self.gt_q = gt_traj[:, 3:]

        gt_v = (self.gt_p[2:] - self.gt_p[:-2]) / (self.gt_ts[2:] - self.gt_ts[:-2])[:, None]
        vs = (self.gt_p[1] - self.gt_p[0]) / (self.gt_ts[1] - self.gt_ts[0])
        gt_v = np.concatenate((vs.reshape((1,3)), gt_v), axis=0)
        vf = (self.gt_p[-1] - self.gt_p[-2]) / (self.gt_ts[-1] - self.gt_ts[-2])
        gt_v = np.concatenate((gt_v, vf.reshape((1,3))), axis=0)
        self.gt_v = gt_v

    def get_datai(self, idx, get_thrust=False):
        ts = self.ts[idx]
        acc = self.accel_raw[idx].reshape((3, 1))
        gyr = self.gyro_raw[idx].reshape((3, 1))
        if get_thrust:
            assert self.accel_raw.shape[0] == self.thrust.shape[0]
            thr = self.thrust[idx].reshape((3, 1))
            return ts, acc, gyr, thr
        else:
            return ts, acc, gyr

    def get_imu_calibration(self):
        imu_calib = {}
        imu_calib["gyro_bias"] = self.gyro_bias
        imu_calib["accel_bias"] = self.accel_bias
        return imu_calib

    def get_groundtruth_pose(self, ts):
        """
        Helper function: This returns the groundtruth pose at the desired time.
        """
        # check if gate detection is available at this time
        is_available = self.gt_ts[0] <= ts <= self.gt_ts[-1]
        if not is_available:
            return False, None, None

        idx_left = np.where(self.gt_ts <= ts)[0][-1]
        idx_right = np.where(self.gt_ts > ts)[0][0]
        interp_gt_ts = self.gt_ts[idx_left : idx_right + 1]
        slerp = Slerp(interp_gt_ts, Rotation.from_quat(self.gt_q[idx_left : idx_right + 1]))
        
        gt_p_interp = interp1d(self.gt_ts, self.gt_p, axis=0)(ts)
        gt_rot_interp = slerp(ts)

        # simulated displacement measurement
        meas = np.zeros((3,4))
        meas[0:3,0:3] = gt_rot_interp.as_matrix()
        meas[:,3] = gt_p_interp
        meas_cov = np.eye(6)
        meas_cov[0:3,0:3] = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # rot
        meas_cov[3:6,3:6] = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # pos
        
        return True, meas, meas_cov

