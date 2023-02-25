"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Util functions for dataset preparation
"""

import numpy as np


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list


def getImuCalib(dtset_name):
    init_calib = {}

    # Blackbird
    init_calib["Blackbird"] = {}
    init_calib["Blackbird"]["gyro_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["Blackbird"]["accel_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["Blackbird"]["T_mat_gyro"] = np.eye(3)
    init_calib["Blackbird"]["T_sens_gyro"] = np.zeros((3, 3))
    init_calib["Blackbird"]["T_mat_accel"] = np.eye(3)
    
    return init_calib[dtset_name]

