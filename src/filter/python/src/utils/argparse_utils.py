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

