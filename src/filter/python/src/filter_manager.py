"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker_runner.py
"""

import os

import numpy as np
import progressbar
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from filter.python.src.data_io import DataIO
from filter.python.src.filter_runner import FilterRunner
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterManager:
    """
    This class is responsible for going through a sequence, feed filter runner and log its result
    """

    def __init__(self, args, sequence_name, outdir):
        self.dataset = args.dataset
        # initialize data IO
        self.input = DataIO()
        self.input.load(args.root_dir, args.dataset, sequence_name)

        # log file initialization
        outfile = os.path.join(outdir, "stamped_traj_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous trajectory log files erased")
        self.traj_outfile = outfile
        self.f_traj_logs = []

        outfile = os.path.join(outdir, "stamped_vel_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous velocity log files erased")
        self.vel_outfile = outfile
        self.f_vel_logs = []

        outfile = os.path.join(outdir, "stamped_bias_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous bias log files erased")
        self.bias_outfile = outfile
        self.f_bias_logs = []
        
        self.log_full_state = args.log_full_state
        if self.log_full_state:
            outfile = os.path.join(outdir, "full_state.txt")
            if os.path.exists(outfile):
                os.remove(outfile)
                logging.warning("previous full state log files erased")
            self.full_state_outfile = outfile
            self.full_state_logs_file = open(self.full_state_outfile, "w")
        
        imu_calibration = self.input.get_imu_calibration()

        filter_tuning = dotdict(
            {
                "g_norm": args.g_norm, # m/s^2
                "sigma_na": args.sigma_na, # m/s^2
                "sigma_ng": args.sigma_ng, # rad/s
                "sigma_nba": args.sigma_nba, # m/s^2/sqrt(s)
                "sigma_nbg": args.sigma_nbg, # rad/s/sqrt(s)
                "init_attitude_sigma": args.init_attitude_sigma,  # rad
                "init_yaw_sigma": args.init_yaw_sigma,  # rad
                "init_vel_sigma": args.init_vel_sigma,  # m/s
                "init_pos_sigma": args.init_pos_sigma,  # m
                "init_bg_sigma": args.init_bg_sigma,  # rad/s
                "init_ba_sigma": args.init_ba_sigma,  # m/s^2
                "use_const_cov": args.use_const_cov,
                "const_cov_val_x": args.const_cov_val_x, # sigma^2
                "const_cov_val_y": args.const_cov_val_y, # sigma^2
                "const_cov_val_z": args.const_cov_val_z, # sigma^2
                "meascov_scale": args.meascov_scale,
                "mahalanobis_factor": args.mahalanobis_factor,
                "mahalanobis_fail_scale": args.mahalanobis_fail_scale
            }
        )

        # FilterRunner object
        if args.initialize_with_offline_calib:
            self.runner = FilterRunner(
                model_path=args.model_path,
                model_param_path=args.model_param_path,
                update_freq=args.update_freq,
                filter_tuning=filter_tuning,
                imu_calib_dic=imu_calibration)
        else:
            self.runner = FilterRunner(
                model_path=args.model_path,
                model_param_path=args.model_param_path,
                update_freq=args.update_freq,
                filter_tuning=filter_tuning)

        # output
        self.log_fullstate_buffer = None

    def __del__(self):
        if self.log_full_state:
            try:
                self.full_state_logs_file.close()
            except Exception as e:
                logging.exception(e)

    def add_data_to_be_logged(self, ts, acc, gyr, with_update, thrust=None):
        # filter data logger
        R_wi, v_wi, p_wi, ba, bg = self.runner.filter.get_evolving_state()

        # transform to body for easier evaluation
        # Adapt this transformation for your case (no need for this transformation in the Blackbird dataset)
        R_ib = np.eye(3)
        p_ib = np.zeros((3,))

        R = R_wi @ R_ib
        v = v_wi
        p = p_wi.flatten() + R_wi @ p_ib
        q = Rotation.from_matrix(R).as_quat()
        ba = ba
        bg = bg

        traj_datapoint = np.array([
            ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
        self.f_traj_logs.append(traj_datapoint)
        bias_datapoint = np.array([ts, bg[0,0], bg[1,0], bg[2,0], ba[0,0], ba[1,0], ba[2,0]])
        self.f_bias_logs.append(bias_datapoint)
        vel_datapoint = np.array([ts, v[0,0], v[1,0], v[2,0]])
        self.f_vel_logs.append(vel_datapoint)

        if self.log_full_state:
            _, Sigma15 = self.runner.filter.get_covariance()
            sigmas = np.diag(Sigma15).reshape(15, 1)
            sigmasyawp = self.runner.filter.get_covariance_yawp().reshape(16, 1)
            inno, meas, pred, meas_sigma, inno_sigma = self.runner.filter.get_debug()
            if not with_update:
                inno *= np.nan
                meas *= np.nan
                pred *= np.nan
                meas_sigma *= np.nan
                inno_sigma *= np.nan

            ts_temp = ts.reshape(1, 1)
            temp = np.concatenate(
                [v, p, ba, bg, acc, gyr, thrust, ts_temp, sigmas, inno, \
                    meas, pred, meas_sigma, inno_sigma, sigmasyawp], axis=0)
            vec_flat = np.append(R.ravel(), temp.ravel(), axis=0)

            if self.log_fullstate_buffer is None:
                self.log_fullstate_buffer = vec_flat
            else:
                self.log_fullstate_buffer = np.vstack((self.log_fullstate_buffer, vec_flat))

            if self.log_fullstate_buffer.shape[0] > 100:
                np.savetxt(self.full_state_logs_file, self.log_fullstate_buffer, delimiter=",")
                self.log_fullstate_buffer = None

    def save_logs(self, save_as_npy):
        logging.info("Saving logs!")
        np.savetxt(self.traj_outfile, np.array(self.f_traj_logs),
                   header="ts x y z qx qy qz qw", fmt="%.12f")
        np.savetxt(self.bias_outfile, np.array(self.f_bias_logs),
                   header="ts bg_x bg_y bg_z ba_x ba_y ba_z", fmt="%.3f")
        np.savetxt(self.vel_outfile, np.array(self.f_vel_logs),
                   header="ts v_x v_y v_z", fmt="%.3f")

        if self.log_full_state:
            np.savetxt(self.full_state_logs_file, self.log_fullstate_buffer, delimiter=",")
            self.log_fullstate_buffer = None
            self.full_state_logs_file.close()

            if save_as_npy:
                # actually convert the .txt to npy to be more storage friendly
                states = np.loadtxt(self.full_state_outfile, delimiter=",")
                np.save(self.full_state_outfile[:-4] + ".npy", states)
                os.remove(self.full_state_outfile)

    def run(self, args):
        # Loop through the entire dataset and feed the data to the imu runner
        n_data = self.input.dataset_size
        for i in progressbar.progressbar(range(n_data), redirect_stdout=True):
        # for i in range(n_data):
            # obtain next raw IMU and thrust measurement from data loader
            ts, acc_raw, gyr_raw, thrust = self.input.get_datai(i, True)
            t_us = from_sec_to_usec(ts)

            if self.runner.filter.initialized:
                did_update = self.runner.on_imu_measurement(t_us, gyr_raw, acc_raw, thrust)
                self.add_data_to_be_logged(
                    ts,
                    self.runner.last_acc,
                    self.runner.last_gyr,
                    with_update=did_update,
                    thrust=self.runner.last_thrust
                )
            else:
                # initialize to gt state R,v,p and offline calib
                if not args.initialize_with_gt:
                    self.runner.on_imu_measurement(t_us, gyr_raw, acc_raw)
                else:
                    if args.initialize_with_offline_calib:
                        init_ba = self.runner.icalib.accelBias
                        init_bg = self.runner.icalib.gyroBias
                    else:
                        init_ba = np.zeros((3, 1))
                        init_bg = np.zeros((3, 1))
                    gt_p = interp1d(self.input.gt_ts, self.input.gt_p, axis=0)(ts)
                    gt_v = interp1d(self.input.gt_ts, self.input.gt_v, axis=0)(ts)
                    gt_rot = Slerp(self.input.gt_ts, Rotation.from_quat(self.input.gt_q))(ts)
                    gt_R = gt_rot.as_matrix()
                    self.runner.filter.initialize_with_state(
                        t_us,
                        gt_R,
                        np.atleast_2d(gt_v).T,
                        np.atleast_2d(gt_p).T,
                        init_ba,
                        init_bg,
                    )
                    self.runner.next_aug_t_us = t_us
                    self.runner.next_interp_t_us = t_us

        self.save_logs(args.save_as_npy)

    def reset_filter_state_from_groundtruth(self, this: FilterRunner):
        """ This reset the filter state from groundtruth state as found in input """
        # compute from ground truth
        inp = self.input
        state = this.filter.state
        gt_ps = []
        gt_Rs = []
        gt_vs = []
        for _, ts_i_us in enumerate(state.si_timestamps_us):
            ts_i = from_usec_to_sec(ts_i_us)
            ps = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_p, axis=0)(ts_i)).T
            gt_ps.append(ps)
            gt_rots = Slerp(self.input.gt_ts, Rotation.from_quat(inp.gt_q))(ts_i)
            gt_Rs.append(gt_rots.as_matrix())
            vs = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_v, axis=0)(ts_i)).T
            gt_vs.append(vs)

        ts = from_usec_to_sec(state.s_timestamp_us)
        gt_p = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_p, axis=0)(ts)).T
        gt_v = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_v, axis=0)(ts)).T
        gt_rot = Slerp(self.input.gt_ts, Rotation.from_quat(inp.gt_q))(ts)
        gt_R = gt_rot.as_matrix()

        this.filter.reset_state_and_covariance(
            gt_Rs, gt_ps, gt_vs, gt_R, gt_v, gt_p, state.s_ba, state.s_bg
        )

    def reset_filter_state_pv(self):
        """ Reset filter states p and v with zeros """
        state = self.runner.filter.state
        ps = []
        vs = []
        for i in state.si_timestamps:
            ps.append(np.zeros((3, 1)))
            vs.append(np.zeros((3, 1)))
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        self.runner.filter.reset_state_and_covariance(
            state.si_Rs, ps, vs, state.s_R, v, p, state.s_ba, state.s_bg
        )

