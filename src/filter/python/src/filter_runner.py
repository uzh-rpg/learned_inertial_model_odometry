"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker.py
"""

import json

from numba import jit
import numpy as np

from filter.python.src.meas_source_network import MeasSourceNetwork
from filter.python.src.net_input_utils import NetInputBuffer, ImuCalib
from filter.python.src.scekf import ImuMSCKF
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.math_utils import mat_exp
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterRunner:
    """
    FilterRunner is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning,
        imu_calib_dic=None,
        force_cpu=False,
    ):
        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["sampling_freq"]
            config_from_network["window_time"] = data_json["window_time"]

        # frequencies and sizes conversion
        self.imu_freq_net = config_from_network.imu_freq_net  # imu frequency as input to the network
        window_size = int(
            (config_from_network.window_time * config_from_network.imu_freq_net) )
        self.net_input_size = window_size

        # EXAMPLE :
        # if using 200 samples with step size 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if using 400 samples with 200 past data and clone_every_n_netimu_sample 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError("window_time cannot be represented by integer number of updates.")
        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(
            config_from_network.imu_freq_net / update_freq
        )  # network inference/filter update interval
        assert (
            config_from_network.imu_freq_net % update_freq == 0
        )  # imu frequency must be a multiple of update frequency
        self.update_distance_num_clone = int(
            config_from_network.window_time * update_freq
        )

        # time
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(1.0 / self.update_freq * 1e6)

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} (samples)"
        )
        logging.info("IMU / Thrust input to the network frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            "Filter update stride state number: %i" % self.update_distance_num_clone
        )
        logging.info(
            f"Interpolating IMU / Thrust measurements every {self.dt_interp_us} [us] for the network input"
        )

        # IMU initial calibration
        self.icalib = ImuCalib()
        self.icalib.from_dic(imu_calib_dic)
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning)

        self.meas_source = MeasSourceNetwork(model_path, force_cpu)

        self.inputs_buffer = NetInputBuffer()

        # This callback is called at first update to initialize the filter
        self.callback_first_update = None

        # keep track of past timestamp and measurement
        self.last_t_us, self.last_acc, self.last_gyr, self.last_thrust = -1, None, None, None
        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False

    # Note, imu meas for the net are calibrated with offline calibration.
    @jit(forceobj=True, parallel=False, cache=False)
    def _get_inputs_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_ts_begin = t_begin_us
        net_ts_end = t_end_us - self.dt_interp_us

        # net_fn are either accel in imu frame or thrusts in imu frame
        net_fn, net_gyr, net_t_us = self.inputs_buffer.get_data_from_to(
            net_ts_begin, net_ts_end
        )

        assert net_gyr.shape[0] == self.net_input_size
        assert net_fn.shape[0] == self.net_input_size
        # get data from filter
        R_oldest_state_wfb, _, _ = self.filter.get_past_state(t_oldest_state_us)  # 3 x 3

        # dynamic rotation integration using filter states
        # Rs_net will contains delta rotation since t_begin_us
        Rs_bofbi = np.zeros((net_t_us.shape[0], 3, 3))  # N x 3 x 3
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_t_us.shape[0]):
            dt_us = net_t_us[j] - net_t_us[j - 1]
            dt = from_usec_to_sec(dt_us)
            dR = mat_exp(net_gyr[j, :].reshape((3, 1)) * dt)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        # find delta rotation index at time ts_oldest_state
        oldest_state_idx_in_net = np.where(net_t_us == t_oldest_state_us)[0][0]

        # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
        # so that Rs_net[idx] = R_oldest_state_wfb
        R_bofboldstate = (
            R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_fn_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_fn)  # N x 3
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N x 3
        net_t_s = from_usec_to_sec(net_t_us)

        return net_gyr_w, net_fn_w, net_t_s

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw, thrust=None):
        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw, thrust)
        else:
            logging.info(f"Initializing filter at time {t_us} [us]")
            if self.icalib:
                logging.info(f"Using bias from initial calibration")
                init_ba = self.icalib.accelBias
                init_bg = self.icalib.gyroBias
                # calibrate raw imu data
                acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                    acc_raw, gyr_raw) 
            else:
                logging.info(f"Using zero bias")
                init_ba = np.zeros((3,1))
                init_bg = np.zeros((3,1))
                acc_biascpst, gyr_biascpst = acc_raw, gyr_raw

            self.filter.initialize(acc_biascpst, t_us, init_ba, init_bg)
            self.next_interp_t_us = t_us
            self.next_aug_t_us = t_us
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, t_us)
            self.next_aug_t_us = t_us + self.dt_update_us
            self.last_t_us, self.last_acc, self.last_gyr = (
                t_us,
                acc_biascpst,
                gyr_biascpst,
            )
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw, thrust=None):
        """
        For new IMU measurement, after the filter has been initialized
        """
        # Eventually calibrate
        if self.icalib:
            # calibrate raw imu data with offline calibation
            # this is used for network feeding
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )

            # calibrate raw imu data with offline calibation scale
            # this is used for the filter. 
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_augmentation_and_update = t_us >= self.next_aug_t_us

        # if augmenting the state, check that we compute interpolated measurement also
        assert (
            do_augmentation_and_update and do_interpolation_of_imu
        ) or not do_augmentation_and_update, (
            "Augmentation and interpolation does not match!"
        )

        # augmentation propagation / propagation
        # propagate at IMU input rate, augmentation propagation depends on t_augmentation_s
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        # Inputs interpolation and data saving for network
        if do_interpolation_of_imu:
            self._add_interpolated_inputs_to_buffer(thrust, gyr_biascpst, t_us)
                
        self.filter.propagate(
            acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us
        )
        # filter update
        did_update = False
        if do_augmentation_and_update:
            did_update = self._process_update(t_us)
            # plan next update/augmentation of state
            self.next_aug_t_us += self.dt_update_us

        # set last value memory to the current one
        self.last_t_us, self.last_acc, self.last_gyr = t_us, acc_biascpst, gyr_biascpst
        self.last_thrust = thrust

        return did_update

    def _process_update(self, t_us):
        logging.debug(f"Upd. @ {t_us} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1 ]
        t_begin_us = t_oldest_state_us
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        # If we do not have enough IMU data yet, just wait for next time
        if t_begin_us < self.inputs_buffer.net_t_us[0]:
            return False
        # initialize with ground truth at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us

        # get measurement from network
        net_gyr_w, net_fn_w, net_t_s = self._get_inputs_samples_for_network(
            t_begin_us, t_oldest_state_us, t_end_us)
        meas, meas_cov = self.meas_source.get_displacement_measurement(
            net_t_s, net_gyr_w, net_fn_w)

        # filter update
        is_available, innovation, jac, noise_mat = \
            self.filter.learnt_model_update(meas, meas_cov, t_oldest_state_us, t_end_us)
        success = False
        if is_available:
            inno = innovation.reshape((3,1))
            success = self.filter.apply_update(inno, jac, noise_mat)

        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        logging.debug(f"marginalize {cut_idx}")
        self.filter.marginalize(cut_idx)
        self.inputs_buffer.throw_data_before(t_begin_us)
        return success

    def _add_interpolated_inputs_to_buffer(self, fn_in, gyr_biascpst, t_us):
        self.inputs_buffer.add_data_interpolated(
            self.last_t_us,
            t_us,
            self.last_gyr,
            gyr_biascpst,
            self.last_thrust,
            fn_in,
            self.next_interp_t_us,
        )

        self.next_interp_t_us += self.dt_interp_us

