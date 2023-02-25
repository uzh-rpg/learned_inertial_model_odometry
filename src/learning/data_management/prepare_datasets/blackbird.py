"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Prepare Blackbird dataset for training, validation, and testing.
"""

import argparse
import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import utils as utils
import rosbag


# coeffs for rotor speeds to thrust conversion
SCALE_THRUST_COEFF = -1.0
MASS = 0.915
C_T = 2.03e-8

# the provided ground truth is the drone body in the NED vicon frame
# rotate to have z upwards
R_w_ned = np.array([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]])
t_w_ned = np.array([0., 0., 0.])

# rotate from body to imu frame
R_b_i = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]])
t_b_i = np.array([0., 0., 0.])

# initial and final times
train_times = {}
train_times['clover/yawForward/maxSpeed5p0'] = [1525745925.0, 1525746016.0]
train_times['halfMoon/yawForward/maxSpeed4p0'] = [1524899787.0, 1524899877.0]
train_times['star/yawForward/maxSpeed5p0'] = [1525686067.0, 1525686107.0]
train_times['egg/yawForward/maxSpeed8p0'] = [1560738505.0, 1560738575.0]
train_times['winter/yawForward/maxSpeed4p0'] = [1525754484.0, 1525754584.0]

val_times = {}
val_times['clover/yawForward/maxSpeed5p0'] = [1525745895.0, 1525745925.0]
val_times['halfMoon/yawForward/maxSpeed4p0'] = [1524899751.0, 1524899787.0]
val_times['star/yawForward/maxSpeed5p0'] = [1525686042.0, 1525686067.0]
val_times['egg/yawForward/maxSpeed8p0'] = [1560738480.0, 1560738505.0]
val_times['winter/yawForward/maxSpeed4p0'] = [1525754454.0, 1525754484.0]

test_times = {}
test_times['clover/yawForward/maxSpeed5p0'] = [1525745865.0, 1525745895.0]
test_times['halfMoon/yawForward/maxSpeed4p0'] = [1524899731.0, 1524899751.0]
test_times['star/yawForward/maxSpeed5p0'] = [1525686026.0, 1525686042.0]
test_times['egg/yawForward/maxSpeed8p0'] = [1560738457.0, 1560738480.0]
test_times['winter/yawForward/maxSpeed4p0'] = [1525754434.0, 1525754454.0]


def prepare_dataset(args):
    dataset_dir = args.dataset_dir

    # read seq names
    seq_names = []
    seq_names.append(utils.get_datalist(os.path.join(dataset_dir, args.data_list)))
    seq_names = [item for sublist in seq_names for item in sublist]

    for idx, seq_name in enumerate(seq_names):
        base_seq_name = os.path.dirname(os.path.dirname(seq_name))
        data_dir = os.path.join(dataset_dir, base_seq_name)
        assert os.path.isdir(data_dir), '%s' % data_dir
        rosbag_fn = os.path.join(data_dir, 'rosbag.bag')

        # Read data
        raw_imu = []  # [ts wx wy wz ax ay az]
        thrusts = []  # [ts 0. 0. mass_normalized_thrust]
        dt = 0.01
        n_discarded_rpm = 0

        imu_topic = '/blackbird/imu'
        rpm_topic = '/blackbird/rotor_rpm'

        print('Reading data from %s' % rosbag_fn)
        with rosbag.Bag(rosbag_fn, 'r') as bag:
            for (topic, msg, ts) in bag.read_messages():
                if topic == imu_topic:
                    imu_i = np.array([
                        msg.header.stamp.to_sec(),
                        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
                    raw_imu.append(imu_i)

                elif topic == rpm_topic:
                    thr_ts = msg.header.stamp.to_sec()

                    t0 = msg.sample_stamp[0].to_sec()
                    t1 = msg.sample_stamp[1].to_sec()
                    t2 = msg.sample_stamp[2].to_sec()
                    t3 = msg.sample_stamp[3].to_sec()
                    t_avg = (t0 + t1 + t2 + t3) / 4.
                    if np.abs(t_avg - t0) > dt or np.abs(t_avg - t1) > dt or \
                            np.abs(t_avg - t2) > dt or np.abs(t_avg - t3) > dt:
                        n_discarded_rpm += 1
                        continue

                    omega1, omega2, omega3, omega4 = msg.rpm[0], msg.rpm[1], msg.rpm[2], msg.rpm[3]
                    sum_sqrt_rpm = omega1 * omega1 + omega2 * omega2 + omega3 * omega3 + omega4 * omega4
                    thr = sum_sqrt_rpm * (C_T / MASS)
                    thr = SCALE_THRUST_COEFF * thr
                    thr_i = np.array([thr_ts, 0., 0., thr])
                    thrusts.append(thr_i)

        print('%d discarded rpm measurements' % n_discarded_rpm)

        raw_imu = np.asarray(raw_imu)
        thrusts = np.asarray(thrusts)

        # load ground truth
        gt_fn = os.path.join(data_dir, 'groundTruthPoses.csv')

        with open(gt_fn, 'rb') as fin:
            data_tmp = np.genfromtxt(fin, delimiter=",")

        data = []
        for data_i in data_tmp:
            ts_i = data_i[0] * 1e-6
            t_i = data_i[1:4]
            R_i = Rotation.from_quat(
                np.array([data_i[5], data_i[6], data_i[7], data_i[4]])).as_matrix()

            # transform to world frame
            R_it = R_w_ned @ R_i
            t_it = t_w_ned + R_w_ned @ t_i

            # transform to imu frame
            t_it = t_it + R_it @ t_b_i
            R_it = R_it @ R_b_i

            q_it = Rotation.from_matrix(R_it).as_quat()
            d = np.array([
                ts_i,
                t_it[0], t_it[1], t_it[2],
                q_it[0], q_it[1], q_it[2], q_it[3]
            ])
            data.append(d)
        data = np.asarray(data)

        # include velocities
        gt_times = data[:, 0]
        gt_pos = data[:, 1:4]

        # compute velocity
        v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
        gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
        gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
        # filter
        gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
        gt_vel_x = gt_vel_x.reshape((-1, 1))
        gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
        gt_vel_y = gt_vel_y.reshape((-1, 1))
        gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
        gt_vel_z = gt_vel_z.reshape((-1, 1))
        gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)

        gt_traj_tmp = np.concatenate((data, gt_vel), axis=1)  # [ts x y z qx qy qz qw vx vy vz]

        # In Blackbird dataset, the sensors measurements are at:
        # 100 Hz IMU meas.
        # 180 Hz RPM meas.
        # 360 Hz Vicon meas.
        # resample imu at exactly 100 Hz
        t_curr = raw_imu[0, 0]
        dt = 0.01
        new_times_imu = [t_curr]
        while t_curr < raw_imu[-1, 0] - dt - 0.001:
            t_curr = t_curr + dt
            new_times_imu.append(t_curr)
        new_times_imu = np.asarray(new_times_imu)
        gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
        accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
        raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

        # We down sample to IMU rate
        times_imu = raw_imu[:, 0]
        # get initial and final times for interpolations
        idx_s = 0
        for ts in times_imu:
            if ts > gt_traj_tmp[0, 0] and ts > thrusts[0, 0]:
                break
            else:
                idx_s = idx_s + 1
        assert idx_s < len(times_imu)

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0] and ts < thrusts[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        times_imu = times_imu[idx_s:idx_e + 1]
        raw_imu = raw_imu[idx_s:idx_e + 1]

        # interpolate ground-truth samples at thrust times
        groundtruth_pos_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu)
        groundtruth_rot_data = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu)
        groundtruth_vel_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu)

        gt_traj = np.concatenate((times_imu.reshape((-1, 1)),
                                  groundtruth_pos_data,
                                  groundtruth_rot_data.as_quat(),
                                  groundtruth_vel_data), axis=1)

        # interpolate thrusts samples at imu times
        thrusts_tmp = interp1d(thrusts[:, 0], thrusts[:, 1:4], axis=0)(times_imu)
        thrusts = thrusts_tmp

        ts = raw_imu[:, 0]

        # Calibrate
        imu_calibrator = utils.getImuCalib("Blackbird")
        b_g = imu_calibrator["gyro_bias"]
        b_a = imu_calibrator["accel_bias"]
        w_calib = raw_imu[:, 1:4].T - b_g[:, None]
        a_calib = raw_imu[:, 4:].T - b_a[:, None]
        calib_imu = np.concatenate((raw_imu[:, 0].reshape((-1, 1)), w_calib.T, a_calib.T), axis=1)

        # sample relevant times
        ts0_train, ts1_train = train_times[base_seq_name]
        idx0_train = np.where(ts > ts0_train)[0][0]
        idx1_train = np.where(ts > ts1_train)[0][0]

        ts0_val, ts1_val = val_times[base_seq_name]
        idx0_val = np.where(ts > ts0_val)[0][0]
        idx1_val = np.where(ts > ts1_val)[0][0]

        ts0_test, ts1_test = test_times[base_seq_name]
        idx0_test = np.where(ts > ts0_test)[0][0]
        idx1_test = np.where(ts > ts1_test)[0][0]

        ts_train = ts[idx0_train:idx1_train]
        raw_imu_train = raw_imu[idx0_train:idx1_train]
        calib_imu_train = calib_imu[idx0_train:idx1_train]
        gt_traj_train = gt_traj[idx0_train:idx1_train]
        thrusts_train = thrusts[idx0_train:idx1_train]
        i_thrusts_train = thrusts_train

        ts_val = ts[idx0_val:idx1_val]
        raw_imu_val = raw_imu[idx0_val:idx1_val]
        calib_imu_val = calib_imu[idx0_val:idx1_val]
        gt_traj_val = gt_traj[idx0_val:idx1_val]
        thrusts_val = thrusts[idx0_val:idx1_val]
        i_thrusts_val = thrusts_val

        ts_test = ts[idx0_test:idx1_test]
        raw_imu_test = raw_imu[idx0_test:idx1_test]
        calib_imu_test = calib_imu[idx0_test:idx1_test]
        gt_traj_test = gt_traj[idx0_test:idx1_test]
        thrusts_test = thrusts[idx0_test:idx1_test]
        i_thrusts_test = thrusts_test

        # Not supported on this branch
        traj_target_oris_from_imu_list = []
        traj_target_oris_from_imu_list.append(gt_traj[0])
        traj_target_oris_from_imu = np.asarray(traj_target_oris_from_imu_list)

        # Save
        # train
        out_dir = os.path.join(data_dir, "imo", "train")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_train)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_train[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_train[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_train[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_train[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_train[:, 1:8])
            traj_target_oris_from_imu_target = \
                f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            thru = f.create_dataset("thrust", data=thrusts_train)
            i_thru = f.create_dataset("i_thrust", data=i_thrusts_train)
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_train, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_train, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_train, fmt='%.12f', header='ts x y z qx qy qz qw')
            np.savetxt(os.path.join(out_dir, "collective_thrust.txt"),
                       np.concatenate((ts_train.reshape((-1, 1)), thrusts_train), axis=1),
                       fmt='%.12f', header='ts thrust [m/s2]')

        print("File data.hdf5 written to " + out_fn)

        # val
        out_dir = os.path.join(data_dir, "imo", "val")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_val)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_val[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_val[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_val[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_val[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_val[:, 1:8])
            traj_target_oris_from_imu_target = \
                f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            thru = f.create_dataset("thrust", data=thrusts_val)
            i_thru = f.create_dataset("i_thrust", data=i_thrusts_val)
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_val, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_val, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_val, fmt='%.12f', header='ts x y z qx qy qz qw')
            np.savetxt(os.path.join(out_dir, "collective_thrust.txt"),
                       np.concatenate((ts_val.reshape((-1, 1)), thrusts_val), axis=1),
                       fmt='%.12f', header='ts thrust [m/s2]')

        print("File data.hdf5 written to " + out_fn)

        # test
        out_dir = os.path.join(data_dir, "imo", "test")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_test)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_test[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_test[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_test[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_test[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_test[:, 1:8])
            traj_target_oris_from_imu_target = \
                f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            thru = f.create_dataset("thrust", data=thrusts_test)
            i_thru = f.create_dataset("i_thrust", data=i_thrusts_test)
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_test, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_test, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_test, fmt='%.12f', header='ts x y z qx qy qz qw')
            np.savetxt(os.path.join(out_dir, "collective_thrust.txt"),
                       np.concatenate((ts_test.reshape((-1, 1)), thrusts_test), axis=1),
                       fmt='%.12f', header='ts thrust [m/s2]')

        print("File data.hdf5 written to " + out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--data_list", type=str)
    parser.add_argument("--save_txt", action="store_true")
    args = parser.parse_args()

    prepare_dataset(args)

