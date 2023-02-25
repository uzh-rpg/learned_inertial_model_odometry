"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Main script. Use it to launch the filter.

Propagation: Imu
Update: Learned Model

Reference: https://github.com/CathIAS/TLIO/blob/master/src/main_filter.py
"""

import argparse
import datetime
import json
import os
from pprint import pprint
from typing_extensions import Required
# silence NumbaPerformanceWarning
import warnings

from numba.core.errors import NumbaPerformanceWarning
import numpy as np

from filter.python.src.filter_manager import FilterManager
from filter.python.src.utils.argparse_utils import add_bool_arg
from filter.python.src.utils.logging import logging
from filter.python.src.utils.profile import profile

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")

    io_groups.add_argument("--root_dir", type=str, help="Path to data directory")
    io_groups.add_argument("--dataset", type=str, required=True)
    io_groups.add_argument("--data_list", type=str, default="test.txt")
    io_groups.add_argument("--checkpoint_fn", type=str, help="path to trained network.", required=True)
    io_groups.add_argument("--model_param_fn", type=str, default='', help="path to .json file")
    io_groups.add_argument("--out_dir", type=str, help="Path to res directory")
    io_groups.add_argument("--save_as_npy", action="store_true")

    # ----------------------- network params -----------------------
    net_groups = parser.add_argument_group("network")

    net_groups.add_argument("--cpu", action="store_true")

    # ----------------------- filter params -----------------------
    filter_group = parser.add_argument_group("filter tuning:")

    filter_group.add_argument("--update_freq", type=float, default=20.0)  # (Hz)

    filter_group.add_argument(
        "--sigma_na", type=float, default=1e-1
    )  # accel noise  m/s^2
    filter_group.add_argument(
        "--sigma_ng", type=float, default=1e-3
    )  # gyro noise  rad/s
    filter_group.add_argument(
        "--sigma_nba", type=float, default=1e-2
    )  # accel bias noise m/s^2/sqrt(s)
    filter_group.add_argument(
        "--sigma_nbg", type=float, default=1e-4
    )  # gyro bias noise rad/s/sqrt(s)

    filter_group.add_argument(
        "--init_attitude_sigma", type=float, default=10.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument(
        "--init_yaw_sigma", type=float, default=10.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)  # m/s
    filter_group.add_argument("--init_pos_sigma", type=float, default=1.0)  # m
    filter_group.add_argument("--init_bg_sigma", type=float, default=1e-4)  # rad/s
    filter_group.add_argument("--init_ba_sigma", type=float, default=1e-4)  # m/s^2
    filter_group.add_argument("--g_norm", type=float, default=9.8082)

    add_bool_arg(
        filter_group, "initialize_with_gt", default=True
    )  # initialize state with gt state
    add_bool_arg(
        filter_group, "initialize_with_offline_calib", default=True
    )  # if True, initialize imu calibration, e.g., biases, with offline calib
    filter_group.add_argument(
        "--mahalanobis_factor", type=float, default=-1.0
    )  # if negative do not do mahalanobis gating test
    filter_group.add_argument(
        "--mahalanobis_fail_scale", type=float, default=0.0
    )  # if nonzero then mahalanobis gating test would scale the covariance by this scale if failed

    add_bool_arg(filter_group, "use_const_cov", default=True)
    filter_group.add_argument(
        "--const_cov_val_x", type=float, default=0.01
    )
    filter_group.add_argument(
        "--const_cov_val_y", type=float, default=0.01
    )
    filter_group.add_argument(
        "--const_cov_val_z", type=float, default=0.01
    )
    filter_group.add_argument("--meascov_scale", type=float, default=1.0)

    add_bool_arg(
        filter_group, "log_full_state", default=False
    )  # log full filter state

    # ----------------------- debug params -----------------------
    debug_group = parser.add_argument_group("debug")
    add_bool_arg(debug_group, "do_profile", default=False, help="Run the profiler")

    args = parser.parse_args()

    np.set_printoptions(linewidth=2000)

    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    # run filter
    with open(os.path.join(args.root_dir, args.dataset, args.data_list)) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    if args.model_param_fn == '':
        model_param_fn = "model_net_parameters.json"
    else:
        model_param_fn = args.model_param_fn
    args.model_path = os.path.join(
        args.out_dir, args.dataset, "checkpoints", "model_net", args.checkpoint_fn)
    args.model_param_path = os.path.join(
        args.out_dir, args.dataset, "checkpoints", "model_net", model_param_fn)

    with profile(filename="./profile.prof", enabled=args.do_profile):
        n_data = len(data_names)
        logging.info("Running on %d sequences" % n_data)
        for i, name in enumerate(data_names):
            logging.info(f"Processing {i+1} / {n_data} dataset {name}")
            
            seq_out_dir = os.path.join(args.out_dir, args.dataset, name, 'pyfilter')
            if not os.path.exists(seq_out_dir):
                os.makedirs(seq_out_dir)

            param_dict = vars(args)
            param_dict["date"] = str(datetime.datetime.now())
            with open(seq_out_dir + "/parameters.json", "w") as parameters_file:
                parameters_file.write(json.dumps(param_dict, indent=4, sort_keys=True))

            try:
                filterManager = FilterManager(args, name, seq_out_dir)
                filterManager.run(args)
            except FileExistsError as e:
                print(e)
                continue
            except OSError as e:
                print(e)
                continue

