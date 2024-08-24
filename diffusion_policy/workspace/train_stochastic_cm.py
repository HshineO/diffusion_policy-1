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

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from diffusion_policy.model.diffusion.stochastic_conditional_unet1d import StochasticConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from diffusion_policy.common.json_logger import JsonLogger

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
# from diffusion_policy.policy.manicm_student_cm_policy import ManiCMStudentPolicy
from diffusion_policy.policy.stochastic_cm_policy import StochasticConsistencyPolicy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)

@torch.no_grad()
def update_ema(target_params: Generator, source_params: Generator, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

# class DDIMSolver:
#     def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
#         # DDIM sampling parameters
#         step_ratio = timesteps // ddim_timesteps
#         self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
#         self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
#         self.ddim_alpha_cumprods_prev = np.asarray(
#             [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
#         )
#         # convert to torch tensors
#         self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
#         self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
#         self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

#     def to(self, device: torch.device) -> "DDIMSolver":
#         self.ddim_timesteps = self.ddim_timesteps.to(device)
#         self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
#         self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
#         return self

#     def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
#                   timestep_index: torch.Tensor) -> torch.Tensor:
#         alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
#         dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
#         x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
#         return x_prev

class TrainStudentWorkspace(BaseWorkspace):
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
        # if cfg.training.resume:
        #     lastest_ckpt_path = pathlib.Path(cfg.teacher_ckpt) # self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)
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

        noise_scheduler = self.model.noise_scheduler
        alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
        sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

        # Teacher DDIM Solver
        solver = copy.deepcopy(self.model.noise_scheduler)

        print(type(self.model.noise_scheduler))

        ############## Load Teacher Model ###############

        teacher_path = cfg.teacher_ckpt 
        teacher = torch.load(teacher_path)

        encoder_state_dict = {
            k[len('obs_encoder.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('obs_encoder.')
        }
        self.model.obs_encoder.load_state_dict(encoder_state_dict)



        # encoder = MultiImageObsEncoder(**cfg.policy.obs_encoder)
        # encoder = torch.load(cfg.teacher_ckpt)
        # print(encoder)

        

        encoder = self.model.obs_encoder
        encoder.to(device)
        # teacher_unet = self.model.model

        # teacher_unet = 

        # print(type(teacher_unet))

        encoder.requires_grad_(False)
        # teacher_unet.requires_grad_(False)

        # compute unet config
        # parse shapes
        action_shape = cfg.shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if cfg.obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * cfg.n_obs_steps

        teacher_unet = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
            down_dims=cfg.policy.down_dims,
            kernel_size=cfg.policy.kernel_size,
            n_groups=cfg.policy.n_groups,
            cond_predict_scale=cfg.policy.cond_predict_scale
        )

        model_state_dict = {
            k[len('model.'):]: v for k, v in teacher['state_dicts']['model'].items()
                if k.startswith('model.')
            }       

        teacher_unet.load_state_dict(model_state_dict)
        teacher_unet.requires_grad_(False)

        unet = StochasticConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim ,
            # diffusion_timestep=cfg.policy.diffusion_timestep * cfg.horizon * cfg.shape_meta.action.shape,
            diffusion_timestep=cfg.policy.diffusion_timestep * 16 * 10,
            down_dims=cfg.policy.down_dims,
            kernel_size=cfg.policy.kernel_size,
            n_groups=cfg.policy.n_groups,
            cond_predict_scale=cfg.policy.cond_predict_scale
        )
        model_state_dict = {
             k[len('model.diffusion_step_encoder.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.diffusion_step_encoder.')
            }
        unet.diffusion_step_encoder.load_state_dict(model_state_dict)

        # @todo:warmup from teacher
        model_state_dict = {
             k[len('model.up_modules.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.up_modules.')
            }
        unet.up_modules.load_state_dict(model_state_dict)
        model_state_dict = {
             k[len('model.mid_modules.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.mid_modules.')
            }
        unet.mid_modules.load_state_dict(model_state_dict)
        model_state_dict = {
             k[len('model.down_modules.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.down_modules.')
            }
        unet.down_modules.load_state_dict(model_state_dict)
        # final_conv
        model_state_dict = {
             k[len('model.final_conv.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.final_conv.')
            }
        unet.final_conv.load_state_dict(model_state_dict)


        # unet.load_state_dict(teacher_unet.state_dict())
        target_unet = StochasticConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
            diffusion_timestep=cfg.policy.diffusion_timestep* 16 * 10,
            down_dims=cfg.policy.down_dims,
            kernel_size=cfg.policy.kernel_size,
            n_groups=cfg.policy.n_groups,
            cond_predict_scale=cfg.policy.cond_predict_scale
        )
        model_state_dict = {
             k[len('model.diffusion_step_encoder.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('model.diffusion_step_encoder.')
            }
        target_unet.diffusion_step_encoder.load_state_dict(model_state_dict)

        # unet = ConditionalUnet1D(*self.model.unet_config)
        # unet.load_state_dict(teacher_unet.state_dict())
        # target_unet = ConditionalUnet1D(*self.model.unet_config)
        # target_unet.load_state_dict(teacher_unet.state_dict())

        self.model.model = unet
        unet = unet.to(device)
        target_unet = target_unet.to(device)
        target_unet.requires_grad_(False)
        encoder.to(device)

        # 没有mask
        # mask_generator = self.model.mask_generator
        # mask_generator.requires_grad_(False)

        normalizer = LinearNormalizer() #self.model.normalizer
        model_state_dict = {
            k[len('normalizer.'):]: v for k, v in teacher['state_dicts']['model'].items()
            if k.startswith('normalizer.')
        }
        normalizer.load_state_dict(model_state_dict)
        normalizer.requires_grad_(False)
        normalizer.to(device)

        self.model.normalizer = normalizer

        teacher_unet.to(device)

        # Also move the alpha and sigma noise schedules to device
        # alpha_schedule = alpha_schedule.to(device)
        # sigma_schedule = sigma_schedule.to(device)
        # solver = solver.to(device)

        optimizer = torch.optim.AdamW(
            # itertools.chain(unet.parameters(), self.model.condition_attention.parameters()),
            unet.parameters(),
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

            
                        # normalize input
                        nobs = normalizer.normalize(batch['obs'])
                        nactions = normalizer['action'].normalize(batch['action'])

                        # if not self.model.use_pc_color:
                        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]
                        
                        batch_size = nactions.shape[0]
                        horizon = nactions.shape[1]

                        # handle different ways of passing observation
                        local_cond = None
                        global_cond = None
                        # attn_global_cond = None
                        trajectory = nactions
                        cond_data = trajectory
                        
                        if self.model.obs_as_global_cond:
                            # reshape B, T, ... to B*T
                            # [batch_size * n_bos_steps, 512, 3], [batch_size * n_bos_steps, 24]
                            this_nobs = dict_apply(nobs, 
                                lambda x: x[:,:self.model.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                            nobs_features = encoder(this_nobs)
                            # output_dim = nobs_features.shape[1]
                            # attn_nobs_features = nobs_features.reshape(-1, self.model.n_obs_steps, output_dim)
                            # attn_nobs_features = self.model.condition_attention(attn_nobs_features, attn_nobs_features).reshape(-1, output_dim)

                            # if "cross_attention" in self.model.condition_type:
                            #     # treat as a sequence
                            #     global_cond = nobs_features.reshape(batch_size, self.model.n_obs_steps, -1)
                            # else:
                            #     # reshape back to B, Do
                            #     global_cond = nobs_features.reshape(batch_size, -1)

                            global_cond = nobs_features.reshape(batch_size, -1)

                                # attn_global_cond = attn_nobs_features.reshape(batch_size, -1)
                            # # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
                            # this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
                            # this_n_point_cloud = this_n_point_cloud[..., :3]

                            trajectory = cond_data.detach()

                        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
                        latents = trajectory

                        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
                        topk = noise_scheduler.config.num_train_timesteps // cfg.policy.num_inference_steps
                        index = torch.randint(0, cfg.policy.num_inference_steps, (batch_size,), device=device).long()

                        solver.set_timesteps(self.model.num_inference_steps,device)

                        start_timesteps = solver.timesteps[index]
                        timesteps = start_timesteps - topk
                        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                        noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                        # teacher network
                        with torch.no_grad():
                            cond_teacher_output = teacher_unet(
                                sample=noisy_model_input, 
                                timestep=start_timesteps, 
                                local_cond=local_cond, 
                                global_cond=global_cond)
                            
                            # t = solver.timesteps[index]
                            # 3. compute previous image: x_t -> x_t-1
                            # start_timesteps is [batch,1]
                            x_prev = cond_teacher_output.clone().to(device) # batch * horizon * action_dim
                            # noise_squence = torch.zeros_like(cond_teacher_output).to(device)
                            T = self.model.num_inference_steps
                            noise_squence = torch.zeros([x_prev.shape[0],T,x_prev.shape[1],x_prev.shape[2]]).to(device) # batch * T * horizon * action_dim

                            # print(x_prev.shape)
                            # print(noise_squence.shape)

                            for i in range(x_prev.shape[0]):
                                t = start_timesteps[i].item()
                                ddpm_output = solver.step(cond_teacher_output[i].unsqueeze(0), t, noisy_model_input[i].unsqueeze(0))
                                x_prev[i] = ddpm_output.prev_sample
                                # print("noise:",ddpm_output.add_noise)
                                noise_squence[i][t] = ddpm_output.add_noise

                        # online network
                        noise_pred = unet(
                            sample=noisy_model_input, 
                            timestep=start_timesteps, 
                            noise_sequence = noise_squence,
                            local_cond=local_cond, 
                            global_cond=global_cond)

                        pred_x_0 = predicted_origin(
                            noise_pred,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule)
                        
                        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
                        
                        # target network
                        noise_squence = torch.zeros([x_prev.shape[0],T,x_prev.shape[1],x_prev.shape[2]]).to(device)
                        with torch.no_grad():
                            target_noise_pred = target_unet(
                                x_prev.float(),
                                timesteps,
                                noise_sequence = noise_squence,
                                local_cond=local_cond, 
                                global_cond=global_cond)
                            pred_x_0 = predicted_origin(
                                target_noise_pred,
                                timesteps,
                                x_prev,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule)
                            target = c_skip * x_prev + c_out * pred_x_0

                        model_pred_mean = model_pred.mean()
                        target_mean = target.mean()
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.training.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_ema(target_unet.parameters(), unet.parameters(), cfg.training.ema_decay)
                        
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
    workspace = TrainStudentWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
