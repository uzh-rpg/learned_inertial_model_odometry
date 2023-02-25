"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/profile.py
"""

import contextlib
import cProfile


@contextlib.contextmanager
def profile(filename, enabled=True):
    if enabled:
        profile = cProfile.Profile()
        profile.enable()
    try:
        yield
    finally:
        if enabled:
            profile.disable()
            profile.dump_stats(filename)


