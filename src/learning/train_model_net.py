"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/train.py
"""

import json
import os
import signal
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.data_management.datasets import ModelDataset
from learning.network.losses import get_error_and_loss
from learning.network.model_factory import get_model
from learning.utils.argparse_utils import arg_conversion
from learning.utils.logging import logging


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(learn_configs, network, data_loader, device):
    """
    Get network status
    """
    errors_all, losses_all = [], []
    
    network.eval()

    for _, (feat, v_init, targ, ts, _, _) in enumerate(data_loader):
        # feat_i = [[feat_gyros], [feat_thrusts]]
        # dims = [batch size, 6, window size]
        # targ = [dp]
        # dims = [batch size, 3]

        feat = feat.to(device)
        v_init = v_init.to(device)
        targ = targ.to(device)

        # get network prediction
        dp = network(feat)

        # compute loss
        errs, loss = get_error_and_loss(dp, targ, learn_configs, device)

        # log
        losses_all.append(torch_to_numpy(loss))
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errors_all.append(errs_norm)

    # save
    losses_all = np.asarray(losses_all)
    errors_all = np.concatenate(errors_all, axis=0)

    attr_dict = {
        "errors": errors_all,
        "losses": losses_all
    }
    
    return attr_dict


def run_train(learn_configs, network, train_loader, device, optimizer):
    """
    Train network for one epoch
    """
    errors_all, losses_all = [], []

    network.train()

    for _, (feat, v_init, targ, ts, _, _) in enumerate(train_loader):
        # feat_i = [[feat_gyros], [feat_thrusts]]
        # dims = [batch size, 6, window size]
        # targ = [dp]
        # dims = [batch size, 3]
        feat = feat.to(device)
        v_init = v_init.to(device)
        targ = targ.to(device)

        optimizer.zero_grad()

        # get network prediction
        dp = network(feat)

        # compute loss
        errs, loss = get_error_and_loss(dp, targ, learn_configs, device)

        # log
        losses_all.append(torch_to_numpy(loss))
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errors_all.append(errs_norm)

        # backprop and optimization
        loss.backward()
        optimizer.step()

    # save
    losses_all = np.asarray(losses_all)
    errors_all = np.concatenate(errors_all, axis=0)

    train_dict = {
        "errors": errors_all,
        "losses": losses_all
    }

    return train_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """
    error = np.mean(attr_dict["errors"])
    loss = np.mean(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_loss_pos/avg", error, epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", loss, epoch)
    logging.info(f"{mode}: average error [m]: {error}")
    logging.info(f"{mode}: average loss: {loss}")

    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1)


def save_model(args, epoch, network, optimizer, interrupt=False):
    if interrupt:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_latest.pt")
    else:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def train(args):
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.train_list is None:
            raise ValueError("train_list must be specified.")
        if args.dataset is None:
            raise ValueError("dataset must be specified.")
        args.out_dir = os.path.join(args.out_dir, args.dataset)
        if args.out_dir != None:
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints"))
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints", "model_net")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints", "model_net"))
            if not os.path.isdir(os.path.join(args.out_dir, "logs")):
                os.makedirs(os.path.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "checkpoints", "model_net", "model_net_parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        if args.val_list is None:
            logging.warning("val_list != specified.")
        if args.continue_from != None:
            if os.path.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info("Training/testing with " + str(data_window_config["sampling_freq"]) + " Hz gyro / thrust data")
    logging.info(
        "Window time: " + str(args.window_time)
        + " [s], " 
        + "Window size: " + str(data_window_config["window_size"])
        + ", "
        + "Window shift time: " + str(data_window_config["window_shift_time"])
        + " [s], "
        + "Window shift size: " + str(data_window_config["window_shift_size"])
    )

    # Network
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )    
    input_dim = args.input_dim
    output_dim = args.output_dim
    network = get_model(input_dim, output_dim).to(
        device
    )
        
    n_params = network.get_num_params()
    params = network.parameters()
    logging.info(f'TCN network loaded to device {device}')
    logging.info(f"Total number of learning parameters: {n_params}")

    # Training / Validation datasets
    train_loader, val_loader = None, None
    start_t = time.time()
    train_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.train_list))
    try:
        train_dataset = ModelDataset(
            args.root_dir, args.dataset, train_list, args, data_window_config, mode="train")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True) #, num_workers=16)
    except OSError as e:
        logging.error(e)
        return
    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(train_dataset)}")

    run_validation = False
    val_list = None
    if args.val_list != '':
        run_validation = True
        val_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.val_list))

    optimizer = torch.optim.Adam(params, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from != None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "inertial_net", "checkpoint_latest.pt")

        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(os.path.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {n_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    attr_dict = get_inference(net_config, network, train_loader, device)
    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    best_loss = np.mean(attr_dict["losses"])
    # run first validation of the full validation set
    if run_validation:
        try:
            val_dataset = ModelDataset(
                args.root_dir, args.dataset, val_list, args, data_window_config, mode="val")
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=16)
        except OSError as e:
            logging.error(e)
            return
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(val_dataset)}")

        val_dict = get_inference(net_config, network, val_loader, device)
        write_summary(summary_writer, val_dict, start_epoch, optimizer, "val")
        best_loss = np.mean(val_dict["losses"])

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    # actual training
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_dict = run_train(net_config, network, train_loader, device, optimizer)
        write_summary(summary_writer, train_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if run_validation:
            # run validation on a random sequence in the validation dataset
            if not args.dataset == 'Blackbird':
                val_sample = np.random.randint(0, len(val_list))
                val_seq = val_list[val_sample]
                logging.info("Running validation on %s" % val_seq)
                try:
                    val_dataset = ModelDataset(
                        args.root_dir, args.dataset, [val_seq], args, data_window_config, mode="val")
                    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)  # , num_workers=16)
                except OSError as e:
                    logging.error(e)
                    return

            val_attr_dict = get_inference(net_config, network, val_loader, device)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            current_loss = np.mean(val_attr_dict["losses"])

            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, network, optimizer)
        else:
            attr_dict = get_inference(net_config, network, train_loader, device)
            current_loss = np.mean(attr_dict["losses"])
            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, network, optimizer)

    logging.info("Training complete.")

    return

