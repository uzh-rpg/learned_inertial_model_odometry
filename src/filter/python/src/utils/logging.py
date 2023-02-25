"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/logging.py
"""

# Copyright 2004-present Facebook. All Rights Reserved.

# Put some color in you day!
import logging
import sys


try:
    import coloredlogs

    coloredlogs.install()
except BaseException:
    pass

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO, # logging.INFO, logging.WARNING, logging.DEBUG
)

