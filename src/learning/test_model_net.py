"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/test.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.data_management.datasets import ModelDataset
from learning.network.losses import get_error_and_loss
from learning.network.model_factory import get_model
from learning.utils.argparse_utils import arg_conversion
from learning.utils.logging import logging


def makeErrorPlot(dp_errors):
    fig = plt.figure("Errors")
    gs = gridspec.GridSpec(3, 1)

    fig.add_subplot(gs[0, 0])
    plt.plot(dp_errors[:,0], label='x')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')
    plt.title('Position errors')

    fig.add_subplot(gs[1, 0])
    plt.plot(dp_errors[:,1], label='y')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')

    fig.add_subplot(gs[2, 0])
    plt.plot(dp_errors[:,2], label='z')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(learn_configs, network, data_loader, device):
    """
    Get network status
    """
    ts_all, targets_all = [], []
    dp_learned_all = []
    errs_all, losses_all = [], []
    
    network.eval()

    for _, (feat, v_init, targ, ts, _, _) in enumerate(data_loader):
        # feat_i = [[feat_gyros], [feat_thrusts]]
        # dims = [batch size, 6, window size]
        # targ = [dp]
        # dims = [batch size, 3]
        feat = feat.to(device)
        targ = targ.to(device)

        # get network prediction
        dp_learned = network(feat)

        # compute loss
        errs, loss = get_error_and_loss(dp_learned, targ, learn_configs, device)
        
        # log
        losses_all.append(torch_to_numpy(loss))
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errs_all.append(errs_norm)

        ts_all.append(torch_to_numpy(ts))
        targets_all.append(torch_to_numpy(targ))

        dp_learned_all.append(torch_to_numpy(dp_learned))

    losses_all = np.asarray(losses_all)
    errs_all = np.concatenate(errs_all, axis=0)

    ts_all = np.concatenate(ts_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    dp_learned_all = np.concatenate(dp_learned_all, axis=0)
        
    attr_dict = {
        "losses": losses_all,
        "errs": errs_all,
        "ts": ts_all,
        "targets": targets_all,
        "dp_learned": dp_learned_all,
        }

    return attr_dict


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list


def sample_dp(ts, dp):
    # dp_i = [t0 t1 dp]
    dp_s = []
        
    dp_i = np.concatenate(([ts[0,0]], [ts[0,-1]], dp[0]))
    dp_s.append(dp_i)

    for tsi, dpi in zip(ts, dp):
        if tsi[0] >= dp_s[-1][1]:
            dp_i = np.concatenate(([tsi[0]], [tsi[-1]], dpi))
            dp_s.append(dp_i)
    dp_s = np.asarray(dp_s)
    return dp_s


def sample_meas(ts, meas):
    # meas_i = [ts x y z]
    meas_s = []
    meas_i = np.concatenate((ts[0].reshape((-1,1)), meas[0].T), axis=1)
    meas_s.append(meas_i)

    for tsi, measi in zip(ts, meas):
        if tsi[0] > meas_s[-1][-1,0]:
            meas_i = np.concatenate((tsi.reshape((-1,1)), measi.T), axis=1)
            meas_s.append(meas_i)
    meas_s = np.concatenate(meas_s, axis=0)
    return meas_s


def test(args):
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.test_list is None:
            raise ValueError("test_list must be specified.")
        if args.dataset is None:
            raise ValueError("dataset must be specified.")
        if args.out_dir is not None:
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    test_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.test_list))
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model_path = os.path.join(args.out_dir, args.dataset, "checkpoints", "model_net", args.model_fn)
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = args.input_dim
    output_dim = args.output_dim
    network = get_model(input_dim, output_dim).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {model_path} loaded to device {device}.")

    # process sequences
    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            seq_dataset = ModelDataset(
                args.root_dir, args.dataset, [data], args, data_window_config, mode="test")
            seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # Obtain outputs
        net_attr_dict = get_inference(net_config, network, seq_loader, device)

        # Print loss infos
        errs_pos = np.mean(net_attr_dict["errs"])
        loss = np.mean(net_attr_dict["losses"])

        logging.info(f"Test: average err [m]: {errs_pos}")
        logging.info(f"Test: average loss: {loss}")
            
        # save displacement related quantities
        ts = net_attr_dict["ts"]
        dp_learned = net_attr_dict["dp_learned"]
        dp_learned_sampled = np.concatenate((ts[:, 0].reshape(-1, 1), ts[:, -1].reshape(-1, 1), dp_learned), axis=1)

        outdir = os.path.join(args.out_dir, args.dataset, data)
        if os.path.exists(outdir) is False:
            os.makedirs(outdir)
        outfile = os.path.join(outdir, "model_net_learnt_predictions.txt")
        np.savetxt(outfile, dp_learned_sampled, fmt="%.12f", header="t0 t1 dpx dpy dpz")

        # save loss
        outfile = os.path.join(outdir, "net_losses.txt")
        np.savetxt(outfile, net_attr_dict["losses"])

        # plotting
        if args.show_plots:
            # compute errors
            dp_targets = net_attr_dict["targets"]
            dp_errs = dp_learned - dp_targets

            makeErrorPlot(dp_errs)

            print("-- dp Errors --")
            print('x')
            print('mean = %.5f' % np.mean(dp_errs[:,0]))
            print('std = %.5f' % np.std(dp_errs[:,0]))
            print('y')
            print('mean = %.5f' % np.mean(dp_errs[:,1]))
            print('std = %.5f' % np.std(dp_errs[:,1]))
            print('z')
            print('mean = %.5f' % np.mean(dp_errs[:,2]))
            print('std = %.5f' % np.std(dp_errs[:,2]))

            if args.show_plots:
                plt.show()
    return

