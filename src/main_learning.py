"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Main script. Use it to load args and launch network training / test.

Reference: https://github.com/CathIAS/TLIO/blob/master/src/main_net.py
"""

import learning.train_model_net as train_model_net
import learning.test_model_net as test_model_net
from learning.utils.argparse_utils import add_bool_arg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ dataset -----------------
    parser.add_argument("--root_dir", type=str, help="Path to data directory")
    parser.add_argument("--train_list", type=str, default="train.txt", help="In folder root_dir.")
    parser.add_argument("--val_list", type=str, default="val.txt", help="In folder root_dir.")
    parser.add_argument("--test_list", type=str, default="test.txt", help="In folder root_dir.")
    parser.add_argument("--out_dir", type=str, help="Path to result directory")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_fn", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)

    # ------------------ architecture and learning params -----------------
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000, help="max num epochs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("--weight_vel_err", type=float, default=1.)
    parser.add_argument("--weight_pos_err", type=float, default=1.)
    parser.add_argument("--loss_type", type=str, default="huber", help="huber,mse")
    parser.add_argument("--huber_vel_loss_delta", type=float, default=0.1, help="value is in [m/s]")
    parser.add_argument("--huber_pos_loss_delta", type=float, default=0.5, help="value is in [m]")

    # ------------------ data perturbation ------------------
    add_bool_arg(parser, "perturb_orientation", default=True)
    parser.add_argument(
        "--perturb_orientation_theta_range", type=float, default=5.0
    )  # degrees
    parser.add_argument(
        "--perturb_orientation_mean", type=float, default=0.0
    )  # degrees
    parser.add_argument(
        "--perturb_orientation_std", type=float, default=2.0
    )  # degrees
    add_bool_arg(parser, "perturb_bias", default=False)
    parser.add_argument("--gyro_bias_perturbation_range", type=float, default=0.01)
    add_bool_arg(parser, "perturb_init_vel", default=True)
    parser.add_argument("--init_vel_sigma", type=float, default=0.3)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument("--imu_freq", type=float, help="imu freq [Hz]")
    parser.add_argument("--sampling_freq", type=float, default=-1.0,
                        help="freq of imu sampling [Hz]. (-1.0 = same as imu_freq)")
    parser.add_argument("--window_time", type=float, default=1.0, help="[s]")
    parser.add_argument("--window_shift_size", type=int, default=1,
                        help="shift size of the input data window")

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plots", default=False)
    add_bool_arg(parser, "show_plots", default=False)

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        train_model_net.train(args)
    elif args.mode == "test":
        test_model_net.test(args)
    else:
        raise ValueError("Undefined mode")

