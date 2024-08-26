if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import itertools
import os

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import torch.nn.functional as F
from typing import Generator
from hydra.core.hydra_config import HydraConfig
# from diffusion_policy_3d.policy.DP_Teacher import DP_Teacher
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from diffusion_policy.policy.stochastic_cm_policy import StochasticConsistencyPolicy

from diffusion_policy.common.json_logger import JsonLogger

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@torch.no_grad()
def update_ema(target_params: Generator, source_params: Generator, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)

class TrainStochasticCPWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self,cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: StochasticConsistencyPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: StochasticConsistencyPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.logging.mode = "offline"
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # False # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        # else:
        #     raise ValueError(f"Training Must Have A Teacher Model !!!!!")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset), print(f"dataset must be BaseImageDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        # Also move the alpha and sigma noise schedules to device
        # alpha_schedule = alpha_schedule.to(device)
        # sigma_schedule = sigma_schedule.to(device)
        # solver = solver.to(device)

        optimizer = torch.optim.AdamW(
            # itertools.chain(unet.parameters(), self.model.condition_attention.parameters()),
            self.model.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)
        
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )


        # self._output_dir = cfg.output_dir

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
    
        if env_runner is not None:
            assert isinstance(env_runner, BaseImageRunner)
        
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # save batch for sampling
        train_sampling_batch = None

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            },
        )

        

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.global_step = 0
        self.epoch = 0
             
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    t_estart = time.time()
                    for batch_idx, batch in enumerate(tepoch):
                        if batch_idx == 0 and cfg.training.debug == True:
                            t_bstart = time.time()
                            print(f"load data time:{t_bstart-t_estart:.3f}")
                        # device transform
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                            
                        # compute loss
                        # raw_loss = self.model.compute_loss(batch)
                        # loss = raw_loss / cfg.training.gradient_accumulate_every
                        # loss.backward()

                        loss = self.model.compute_loss()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), cfg.training.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_ema(self.model.target_unet.parameters(), self.model.model.parameters(), cfg.training.ema_decay)
                        
                        # logging
                        raw_loss_cpu = loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        loss_dict = {'bc_loss': loss.item()}
                        step_log.update(loss_dict)
                        json_logger.log(step_log)


                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                
                # ========= eval for this epoch ==========
                policy = self.model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)
                    
                # run validation
                t_time = 0
                count = 0
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        val_mse_error = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            t_estart = time.time()
                            for batch_idx, batch in enumerate(tepoch):
                                if batch_idx == 0 and cfg.training.debug == True:
                                    t_bstart = time.time()
                                    print(f"load data time:{t_bstart-t_estart:.3f}")
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, loss_dict = self.model.compute_bc_loss(batch)
                                if (self.epoch % cfg.training.val_sample_every) == 0: # 每多少个epoch进行一次验证集的采样验证，生成action计算mse
                                    obs_dict = batch['obs']
                                    gt_action = batch['action']

                                            
                                    start_time = time.time()
                                    # Student Consistency Inference
                                    result = policy.predict_action(obs_dict)
                                    t = time.time() - start_time

                                    t_time += t
                                    count += 1
                                            

                                    pred_action = result['action_pred']
                                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)


                                    val_losses.append(loss)
                                    val_mse_error.append(mse.item())
                                    del obs_dict
                                    del gt_action
                                    del result
                                    del pred_action
                                    del mse
                                        
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                        if len(val_mse_error) > 0:
                            val_mse_error = torch.mean(torch.tensor(val_mse_error)).item()
                            step_log['val_mse_error'] = val_mse_error

                            val_avg_inference_time = t_time / count
                            step_log['val_avg_inference_time'] = val_avg_inference_time


                
                # run diffusion sampling on a training batch
                t_time = 0
                count = 0
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()
                

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                del step_log

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '../config'))
)
def main(cfg):
    workspace = TrainStochasticCPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
