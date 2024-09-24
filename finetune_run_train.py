import datetime
import os
import os.path
import gc
from itertools import chain
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils1 as utils
from model import SEDD
from model.ema import ExponentialMovingAverage

from PL_DeepSTARR import *


torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    # print (rank)
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    # load the pre-trained model based on EvoAug
    pre_trained_checkpoint_dir = "/home/ykang/d3_test/D3_DNA_Discrete_Diffusion/Training_and_Sampling/exp_local/deepstarr/EvoAug_noRC_25percent/2024.09.03/134709/checkpoints/checkpoint_9.pth"
    # state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    state = utils.restore_checkpoint(pre_trained_checkpoint_dir, state, device)
    # initial_step = int(state['step'])
    mprint(f"load pre-trained model from: {pre_trained_checkpoint_dir}")
    mprint(f"original state['step']: {int(state['step'])}")
    # reset it to 0
    state['step'] = 0
    initial_step = int(state['step'])
    mprint(f"reset step to: {initial_step}")

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']

        batch = next(train_iter)#.to(device)

        #Select lines for input and target seclection based on choice of dataset
        inputs, target = batch #Specific for DeepSTARR and MPRA

        # take a look at inputs, target of the current batch
        # mprint("step: %d, input shape: %s, target shape: %s" % (step, str(inputs.shape), str(target.shape)))

        # seq_one_hot = batch[:, :, :4] #Specific for Promoter
        # inputs = torch.argmax(seq_one_hot, dim=-1) #Specific for Promoter
        # target = batch[:, :, 4:5] #Specific for Promoter
        
        inputs, target = inputs.to(device), target.to(device)
        loss = train_step_fn(state, inputs, target)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                eval_batch = next(eval_iter)
                eval_inputs, eval_target = eval_batch
                eval_inputs, eval_target = eval_inputs.to(device), eval_target.to(device)
                eval_loss = eval_step_fn(state, eval_inputs, eval_target)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))


            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'pretrained_ckpt9_finetune_checkpoint_{save_step}.pth'), state)

                # Save EMA model
                if cfg.training.snapshot_sampling:
                    # mprint(f"Generating text at step: {step}")

                    # this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    # utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    ema.restore(score_model.parameters())

                    # np.savez(os.path.join(this_sample_dir, f"sample_{rank}.npz",), val_pred_seqs.cpu())

                    dist.barrier()
