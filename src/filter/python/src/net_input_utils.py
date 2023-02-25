"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_calib.py
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_buffer.py
"""

import numpy as np
from scipy.interpolate import interp1d


class ImuCalib:
    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.zeros((3,3))
        self.accelBias = np.zeros((3,1))
        self.gyroBias = np.zeros((3,1))

    # @ToDo: Extend to scale factors and g-sensitivity 
    def from_dic(self, imu_calib_dic):
        self.gyroBias = imu_calib_dic["gyro_bias"].reshape((3,1))
        self.accelBias = imu_calib_dic["accel_bias"].reshape((3,1))

    def calibrate_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc) - self.accelBias
        gyr_cal = (
            np.dot(self.gyroScaleInv, gyr)
            - np.dot(self.gyroGSense, acc)
            - self.gyroBias
        )
        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc)
        gyr_cal = np.dot(self.gyroScaleInv, gyr) - np.dot(self.gyroGSense, acc)
        return acc_cal, gyr_cal


class NetInputBuffer:
    """ This is a buffer for interpolated net input data data."""

    def __init__(self):
        self.net_t_us = np.array([])
        self.net_fn = np.array([])  # mass normalized force
        self.net_gyr = np.array([])

    def add_data_interpolated(
        self, last_t_us, t_us, last_gyr, gyr, last_fn, fn, requested_interpolated_t_us
    ):
        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)

        if last_t_us < 0:
            fn_interp = fn.T
            gyr_interp = gyr.T
        else:
            try:
                fn_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_fn.T, fn.T]), axis=0)(requested_interpolated_t_us)
                gyr_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_gyr.T, gyr.T]), axis=0)(requested_interpolated_t_us)
            except ValueError as e:
                print(
                    f"Trying to do interpolation at {requested_interpolated_t_us} between {last_t_us} and {t_us}"
                )
                raise e
        self._add_data(requested_interpolated_t_us, fn_interp, gyr_interp)

    def _add_data(self, t_us, fn, gyr):
        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert (
                t_us > self.net_t_us[-1]
            ), f"trying to insert a data at time {t_us} which is before {self.net_t_us[-1]}"

        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_fn = np.append(self.net_fn, fn).reshape(-1, 3)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)

    # get network data by input size, extract from the latest
    def get_last_k_data(self, size):
        net_fn = self.net_fn[-size:, :]
        net_gyr = self.net_gyr[-size:, :]
        net_t_us = self.net_t_us[-size:]
        return net_fn, net_gyr, net_t_us

    # get network data from beginning and end timestamps
    def get_data_from_to(self, t_begin_us: int, t_us_end: int):
        """ This returns all the data from ts_begin to ts_end """
        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        end_idx = np.where(self.net_t_us == t_us_end)[0][0]
        net_fn = self.net_fn[begin_idx : end_idx + 1, :]
        net_gyr = self.net_gyr[begin_idx : end_idx + 1, :]
        net_t_us = self.net_t_us[begin_idx : end_idx + 1]
        return net_fn, net_gyr, net_t_us

    def throw_data_before(self, t_begin_us: int):
        """ throw away data with timestamp before ts_begin
        """
        assert isinstance(t_begin_us, int)
        begin_idx = np.where(self.net_t_us == t_begin_us)[0][0]
        self.net_fn = self.net_fn[begin_idx:, :]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        return self.net_t_us.shape[0]

    def debugstring(self, query_us):
        print(f"min:{self.net_t_us[0]}")
        print(f"max:{self.net_t_us[-1]}")
        print(f"que:{query_us}")
        print(f"all:{self.net_t_us}")

