"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Math utils
"""

import numpy as np

# Source: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py
def associateTimestamps(first_stamps, second_stamps, offset=0.0, max_difference=1.0):
    """
    associate timestamps
    first_stamps, second_stamps: list of timestamps to associate
    Output:
    sorted list of matches (match_first_idx, match_second_idx)
    """
    potential_matches = [(abs(a - (b + offset)), idx_a, idx_b)
                         for idx_a, a in enumerate(first_stamps)
                         for idx_b, b in enumerate(second_stamps)
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()  # prefer the closest

    matches = []
    first_idxes = list(range(len(first_stamps)))
    second_idxes = list(range(len(second_stamps)))
    for diff, idx_a, idx_b in potential_matches:
        if idx_a in first_idxes and idx_b in second_idxes:
            first_idxes.remove(idx_a)
            second_idxes.remove(idx_b)
            matches.append((int(idx_a), int(idx_b)))

    matches.sort()
    return matches


def hat(v):
    v = v.flatten()
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R


def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")
    angle = np.linalg.norm(omega)

    # Near phi==0, use first order Taylor expansion
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

