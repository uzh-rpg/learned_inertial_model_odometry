"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""


import numpy as np
from scipy.spatial.transform import Rotation


# From seconds to microseconds
def from_sec_to_usec(t_sec):
    return int(t_sec * 1e6)


# From microseconds to seconds
def from_usec_to_sec(t_usec):
    return t_usec * 1e-6


# get trajectory in numpy format ready to be saved.
# [in] ts_list = list of timestamps
# [in] ori_list = list of rotation matrices
# [in] pos_list = list of positions
def getNumpyTraj(ts_list, ori_list, pos_list):
    assert len(ts_list) == len(ori_list) == len(pos_list)

    data = []
    for i, ts in enumerate(ts_list):
        R = ori_list[i]
        q = Rotation.from_matrix(R).as_quat()
        p = pos_list[i]
        datapoint = np.array([
            ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
        data.append(datapoint)

    return np.asarray(data)

