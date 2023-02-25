"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/dataloader/dataset_fb.py
"""

from abc import ABC, abstractmethod
import os
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

import learning.utils.pose as pose


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass


class ModelSequence(CompiledSequence):
    def __init__(self, bsdir, dtset_fn, seq_fn, **kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.features,
            self.targets,
            self.ts,
            self.gyro_raw,
            self.thrust,
            self.feat,
            self.traj_target
        ) = (None, None, None, None, None, None, None, None)

        data_path = os.path.join(bsdir, dtset_fn, seq_fn)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        with h5py.File(os.path.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            thrust = np.copy(f["thrust"])
            i_thrust = np.copy(f["i_thrust"])  # in imu frame
            traj_target = np.copy(f["traj_target"])

        assert thrust.shape[0] == gyro_calib.shape[0], \
            "Make sure that initial and final times correspond to first and last thrust measurement in %s!" % data_path

        # rotate to world frame
        w_gyro_calib = np.array([pose.xyzwQuatToMat(T_wi[3:]) @ w_i for T_wi, w_i in zip(traj_target, gyro_calib)])
        w_thrust = np.array([pose.xyzwQuatToMat(T_wi[3:]) @ t_i for T_wi, t_i in zip(traj_target, i_thrust)])

        self.ts = ts
        self.gyro_raw = gyro_raw
        self.thrust = thrust
        self.feat = np.concatenate([w_gyro_calib, w_thrust], axis=1)
        self.traj_target = traj_target

    def get_feature(self):
        return self.feat

    def get_target(self):
        return self.traj_target

    # Auxiliary quantities, not used for training.
    def get_aux(self):
        return self.ts, self.gyro_raw, self.thrust


class ModelDataset(Dataset):
    def __init__(self, root_dir, dataset_fn, data_list, args, data_window_config, **kwargs):
        super(ModelDataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.8082])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
        self.perturb_orientation_mean = args.perturb_orientation_mean
        self.perturb_orientation_std = args.perturb_orientation_std
        self.perturb_bias = args.perturb_bias
        self.gyro_bias_perturbation_range = args.gyro_bias_perturbation_range
        self.perturb_init_vel = args.perturb_init_vel
        self.init_vel_sigma = args.init_vel_sigma
        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        # index_map = [[seq_id, index of the last datapoint in the window], ...]
        self.index_map = []
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(root_dir, dataset_fn, data_list[i], **kwargs)
            # feat = np.array([[wx, wy, wz, thrx, thry, thrz], ...])
            # targ = np.array([[x, y, z, qx, qy, qz, qw], ...])
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), 
                    N,
                    self.window_shift_size) 
                ]

            times, raw_gyro_meas, thrusts = seq.get_aux()
            self.ts.append(times)

            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.thrusts.append(thrusts)
                
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        idxs = frame_id - self.window_size * self.sampling_factor
        idxe = frame_id
        indices = range(idxs, idxe, self.sampling_factor)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]

        # vel
        if idxs < 0:
            t0m1 = self.ts[seq_id][idxs]
            pw0m1 = self.targets[seq_id][idxs][0:3]
        else:
            t0m1 = self.ts[seq_id][idxs-1]
            pw0m1 = self.targets[seq_id][idxs-1][0:3]
        t0p1 = self.ts[seq_id][idxs+1]
        pw0p1 = self.targets[seq_id][idxs+1][0:3]
        vw0 = (pw0p1 - pw0m1) / (t0p1 - t0m1)

        # pos
        pw0 = self.targets[seq_id][idxs][0:3]
        pw1 = self.targets[seq_id][idxe][0:3]
        targ = pw1 - pw0

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        thrust_i = np.zeros((3,))
        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            thrust_i = self.thrusts[seq_id][indices]
            
        if self.mode == "train":
            # perturb biases
            if self.perturb_bias:
                random_bias = [
                    (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
                    (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
                    (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
                ]
                feat[:, 0] = feat[:, 0] + random_bias[0]
                feat[:, 1] = feat[:, 1] + random_bias[1]
                feat[:, 2] = feat[:, 2] + random_bias[2]

            if self.perturb_orientation:
                vec_rand = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
                vec_rand = vec_rand / np.linalg.norm(vec_rand)

                theta_rand = (
                        random.random() * np.pi * self.perturb_orientation_theta_range / 180.0)
                # theta_deg = np.random.normal(self.perturb_orientation_mean, self.perturb_orientation_std)
                # theta_rand = theta_deg * np.pi / 180.0

                R_mat = pose.fromAngleAxisToRotMat(theta_rand, vec_rand)

                feat[:, 0:3] = np.matmul(R_mat, feat[:, 0:3].T).T
                feat[:, 3:6] = np.matmul(R_mat, feat[:, 3:6].T).T

            # perturb initial velocity
            if self.perturb_init_vel:
                dv = np.array([
                    np.random.normal(scale=self.init_vel_sigma),
                    np.random.normal(scale=self.init_vel_sigma),
                    np.random.normal(scale=2*self.init_vel_sigma)])
                vw0 = vw0 + dv

        return feat.astype(np.float32).T, vw0.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, thrust_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

