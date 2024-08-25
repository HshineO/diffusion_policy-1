from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_lcm import LCMScheduler  #DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  #DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from diffusion_policy.model.diffusion.stochastic_conditional_unet1d import StochasticConditionalUnet1D

from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
import re
import copy

class StochasticConsistencyPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            scheduler: LCMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_timesteps = 1,
            num_inference_steps=None,
            num_train_steps = 100,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            diffusion_noise_embed_output_dim=2048,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            teacher_path=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.action_dim = action_dim

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        teacher_unet = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        online_unet = StochasticConditionalUnet1D(
            input_dim = input_dim, local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            diffusion_noise_embed_input_dim = num_train_steps*horizon*n_action_steps,
            diffusion_noise_embed_output_dim = diffusion_noise_embed_output_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        target_unet = StochasticConditionalUnet1D(
            input_dim = input_dim, local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            diffusion_noise_embed_input_dim = num_train_steps*horizon*n_action_steps,
            diffusion_noise_embed_output_dim = diffusion_noise_embed_output_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        #### load models
        teacher_ckpt = torch.load(teacher_path)
        # load teacher unet
        teacher_unet.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.'))
        # load obs_encoder
        obs_encoder.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'obs_encoder\.'))

        # load online unet
        online_unet.diffusion_step_encoder.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.diffusion_step_encoder\.'))
        online_unet.down_modules.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.down_modules\.'))
        online_unet.up_modules.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.up_modules\.'))
        online_unet.mid_modules.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.mid_modules\.'))
        online_unet.final_conv.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'model\.final_conv\.'))
        
        obs_encoder.requires_grad_(False)
        teacher_unet.requires_grad_(False)
        target_unet.requires_grad_(False)

        self.obs_encoder = obs_encoder.to(self.device)
        self.teacher = teacher_unet.to(self.device)
        self.model = online_unet.to(self.device)
        self.target_unet = target_unet.to(self.device)
        
        self.noise_scheduler = noise_scheduler
        self.scheduler = scheduler

        self.DDPMSolver = copy.deepcopy(noise_scheduler)

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            state_dict_to_model(teacher_ckpt,pattern = r'normalizer\.'))
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # DDIM step
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.num_train_steps = num_train_steps
        
        # LCM step
        self.num_inference_timesteps = num_inference_timesteps

        
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        # scheduler = self.noise_scheduler
        

        # get DDPM previous Noise
        ddpm = self.noise_scheduler
        ddpm.set_timesteps(self.num_inference_steps)

        # previous_noise = torch.zeros(condition_data.shape[0],self.num_train_steps)
        T = self.num_inference_steps
        previous_noise = torch.zeros([condition_data.shape[0],T,self.horizon,self.action_dim]).to(self.device) 
        # batch * T * horizon * action_dim

        previous_noise = ddpm.get_variance_sequence(previous_noise.shape).to(self.device)

        # get gaussion noise x_T
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)
    
        latents = trajectory * self.scheduler.init_noise_sigma
        # set step values
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(latents.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}

        for i, t in enumerate(timesteps):
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model(
                sample=latent_model_input,
                timestep=t,
                noise_sequence = previous_noise,
                local_cond=local_cond,
                global_cond=global_cond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
        trajectory = latents     

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    ## @todo: now training is in workspace 
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
            trajectory = cond_data.detach()
        # else:
        #     # reshape B, T, ... to B*T
        #     this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        #     nobs_features = self.obs_encoder(this_nobs)
        #     # reshape back to B, T, Do
        #     nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        #     cond_data = torch.cat([nactions, nobs_features], dim=-1)
        #     trajectory = cond_data.detach()

        # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        latents = trajectory

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.noise_scheduler.config.num_train_timesteps // self.num_inference_steps
        index = torch.randint(0, self.num_inference_steps, (batch_size,), device=trajectory.device).long()
        
        self.DDPMSolver.set_timesteps(self.num_inference_steps,self.device)

        start_timesteps = self.DDPMSolver.timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)
        
        # teacher network
        with torch.no_grad():
            cond_teacher_output = self.teacher(
                                sample=noisy_model_input, 
                                timestep=start_timesteps, 
                                local_cond=local_cond, 
                                global_cond=global_cond)
                            
            # t = solver.timesteps[index]
            # 3. compute previous image: x_t -> x_t-1
            # start_timesteps is [batch,1]
            x_prev = cond_teacher_output.clone().to(self.device) # batch * horizon * action_dim
                            # noise_squence = torch.zeros_like(cond_teacher_output).to(device)
            T = self.model.num_inference_steps
            noise_squence = torch.zeros([x_prev.shape[0],T,x_prev.shape[1],x_prev.shape[2]]).to(self.device) # batch * T * horizon * action_dim

            for i in range(x_prev.shape[0]):
                t = start_timesteps[i].item()
                ddpm_output = self.DDPMSolver.step(cond_teacher_output[i].unsqueeze(0), t, noisy_model_input[i].unsqueeze(0))
                x_prev[i] = ddpm_output.prev_sample
                # print("noise:",ddpm_output.add_noise)
                noise_squence[i][t] = ddpm_output.add_noise       

        # online network
        noise_pred = self.model(
                            sample=noisy_model_input, 
                            timestep=start_timesteps, 
                            noise_sequence = noise_squence,
                            local_cond=local_cond, 
                            global_cond=global_cond)
        pred_x_0 = predicted_origin(
                            noise_pred,
                            start_timesteps,
                            noisy_model_input,
                            self.noise_scheduler.config.prediction_type)
                            #alpha_schedule,
                            #sigma_schedule)
                        
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        # target network
        noise_squence = torch.zeros([x_prev.shape[0],T,x_prev.shape[1],x_prev.shape[2]]).to(self.device)
        with torch.no_grad():
            target_noise_pred = self.target_unet(
                                x_prev.float(),
                                timesteps,
                                noise_sequence = noise_squence,
                                local_cond=local_cond, 
                                global_cond=global_cond)
            pred_x_0 = predicted_origin(
                                target_noise_pred,
                                timesteps,
                                x_prev,
                                self.noise_scheduler.config.prediction_type)
                                # alpha_schedule,
                                # sigma_schedule)
            target = c_skip * x_prev + c_out * pred_x_0

        model_pred_mean = model_pred.mean()
        target_mean = target.mean()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def compute_bc_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        T = self.num_inference_steps
        previous_noise = torch.zeros([noisy_trajectory.shape[0],T,self.horizon,self.action_dim]).to(self.device) 
        # batch * T * horizon * action_dim

        previous_noise = self.noise_scheduler.get_variance_sequence(previous_noise.shape).to(self.device)
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, previous_noise,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }
        return loss,loss_dict



def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)
    MODEL_PREFIX_LENGTH = len(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[MODEL_PREFIX_LENGTH:]  
            new_state_dict[new_k] = v
    return new_state_dict

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
        alphas: torch.Tensor=None,
        sigmas: torch.Tensor=None
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