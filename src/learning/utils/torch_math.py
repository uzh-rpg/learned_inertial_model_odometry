"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Torch math utils
"""

import torch


def getRotationAngle(R):
    return torch.arccos((torch.trace(R) - 1 - 1e-5) / 2)


def hat(v, device):
    v = v.flatten()
    R = torch.tensor([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]]).to(device)
    return R


def matExp(omega, device):
    I = torch.eye(3).to(device)
    
    angle = torch.linalg.norm(omega)
    # Near angle==0, use first order Taylor expansion
    if angle < 1e-10:
        return I + hat(omega, device)

    axis = omega / angle
    s = torch.sin(angle)
    c = torch.cos(angle)
    return c * I + (1 - c) * torch.outer(axis, axis) + s * hat(axis, device)

