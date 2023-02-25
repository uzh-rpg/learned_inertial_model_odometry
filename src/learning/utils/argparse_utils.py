"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/argparse_utils.py
"""

import argparse

import numpy as np

def add_bool_arg(parser, name, default=False, **kwargs):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name,
        dest=name,
        action="store_true",
        help="Default: " + ("Enabled" if default else "Disabled"),
    )
    group.add_argument("--no-" + name, dest=name, action="store_false", **kwargs)
    parser.set_defaults(**{name: default})


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    # Detected sampling freq different from imu freq
    imu_freq = args.imu_freq
    if args.sampling_freq > 0.0:
        if not (imu_freq / args.sampling_freq).is_integer():
            raise ValueError("sampling_freq must be divisible by imu_freq.")
        else:
            sampling_factor = int(imu_freq / args.sampling_freq)
            sampling_freq = args.sampling_freq
    else:
        sampling_factor = 1
        sampling_freq = imu_freq

    window_size = int((args.window_time * imu_freq) / sampling_factor)
    if window_size < 1:
        raise ValueError("window_size less than 1. Try to increase sampling_freq.")
    
    window_shift_size = args.window_shift_size
    window_shift_time = 1. / imu_freq * sampling_factor

    data_window_config = dict(
        [("sampling_factor", sampling_factor),
        ("sampling_freq", sampling_freq),
        ("imu_freq", imu_freq),
        ("window_size", window_size),
        ("window_shift_size", window_shift_size),
        ("window_shift_time", window_shift_time)]
    )
    net_config = {
        "net_type": 'model',
        "weight_vel_err": args.weight_vel_err,
        "weight_pos_err": args.weight_pos_err,
        "loss_type": args.loss_type,
        "huber_vel_loss_delta": args.huber_vel_loss_delta * args.weight_vel_err,
        "huber_pos_loss_delta": args.huber_pos_loss_delta * args.weight_pos_err
    }

    return data_window_config, net_config

