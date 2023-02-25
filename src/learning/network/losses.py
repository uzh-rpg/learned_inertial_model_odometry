"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/losses.py
"""

import numpy as np
import torch
from torch import nn


"""
Inputs:
- dp = [dpi]. Integrated accelerations to get positions. Dims: (batch_size x 3)
    dpi = [dpxi, dpyi, dpzi]
- targets = [targ_i]. Dims: (batch_size x 3 x 2)
    targ_i = [dp_i_i+1]
- learn_configs
- device

Outputs: 
- position errors [errs_pos]: (batch_size x 3,)
- position loss [loss_pos]: (scalar)
"""
def get_error_and_loss(dp, dp_targets, learn_configs, device):
    weight = torch.tensor(learn_configs["weight_pos_err"], dtype=torch.float32)
    weight = weight.to(device)

    # compute errors
    errs = dp - dp_targets

    # compute losses
    loss_type = learn_configs["loss_type"]
    if loss_type == "huber":
        huber_loss = nn.HuberLoss(delta=learn_configs["huber_pos_loss_delta"])
        loss = weight * huber_loss(dp, dp_targets)

    elif loss_type == "mse":
        loss = weight * torch.mean((errs).pow(2))

    else:
        AssertionError("Unknown loss function!")

    return errs, loss


"""
Inputs:
- dv = [dvi]. Integrated accelerations to get velocities. Dims: (batch_size x 3)
    dvi = [dvxi, dvyi, dvzi]
- dp = [dpi]. Integrated accelerations to get positions. Dims: (batch_size x 3)
    dpi = [dpxi, dpyi, dpzi]
- targets = [targ_i]. Dims: (batch_size x 3 x 2)
    targ_i = [dv_i_i+1, dp_i_i+1]
- learn_configs
- device

Outputs: 
- velocity errors [errs_vel]: (batch_size x 3,) 
- position errors [errs_pos]: (batch_size x 3,)
- velocity loss [loss_vel]: (scalar) 
- position loss [loss_pos]: (scalar)
"""
def get_errors_and_losses(dv, dp, targets, learn_configs, device):
    weight_vel_loss = learn_configs["weight_vel_err"]
    weight_pos_loss = learn_configs["weight_pos_err"]

    weights_arr = np.array([weight_vel_loss, weight_pos_loss])
    weights = torch.from_numpy(weights_arr).to(torch.float32)
    weights = weights.to(device)

    dv_targets = targets[:, :, 0]
    dp_targets = targets[:, :, 1]

    # compute errors
    errs_vel = dv - dv_targets
    errs_pos = dp - dp_targets

    # compute losses
    loss_type = learn_configs["loss_type"]
    if loss_type == "huber":
        huber_loss_vel = nn.HuberLoss(delta=learn_configs["huber_vel_loss_delta"])
        huber_loss_pos = nn.HuberLoss(delta=learn_configs["huber_pos_loss_delta"])

        loss_vel = weights[0] * huber_loss_vel(dv, dv_targets)
        loss_pos = weights[1] * huber_loss_pos(dp, dp_targets)

    elif loss_type == "mse":
        loss_vel = weights[0] * torch.mean((errs_vel).pow(2))
        loss_pos = weights[1] * torch.mean((errs_pos).pow(2))

    else:
        AssertionError("Unknown loss function!")

    return errs_vel, errs_pos, loss_vel, loss_pos

