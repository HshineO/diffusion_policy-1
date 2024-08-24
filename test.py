from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet

from diffusion_policy.policy.manicm_student_cm_policy import ManiCMStudentPolicy
import torch
#   noise_scheduler:
#     _target_: diffusers.schedulers.scheduling_ddim.DDIMScheduler
#     num_train_timesteps: 100
#     beta_start: 0.0001
#     beta_end: 0.02
#     # beta_schedule is important
#     # this is the best we found
#     beta_schedule: squaredcos_cap_v2
#     clip_sample: True
#     set_alpha_to_one: True
#     steps_offset: 0
#     prediction_type: sample # or epsilon

#   scheduler:
#     _target_: diffusers.schedulers.scheduling_lcm.LCMScheduler
#     num_train_timesteps: 100
#     beta_start: 0.0001
#     beta_end: 0.02
#     beta_schedule: squaredcos_cap_v2
#     clip_sample: True
#     set_alpha_to_one: True
#     steps_offset: 0
#     prediction_type: sample

# ddim = DDIMScheduler(100,beta_start= 0.001,beta_end=0.02,beta_schedule="squaredcos_cap_v2",
#               clip_sample= True,prediction_type="sample")
# ddim.set_timesteps(10)

# # print(ddim.timesteps)

# ddpm = DDPMScheduler(10,beta_start= 0.001,beta_end=0.02,beta_schedule="squaredcos_cap_v2",
#               clip_sample= True,prediction_type="sample")
# ddpm.set_timesteps(10)

# print(ddpm.timesteps)

# # start_timesteps = ddim.timesteps*0.3
# # print(type(start_timesteps))
# # print(len(start_timesteps.shape))

# # start_timesteps = start_timesteps[None]
# # print(start_timesteps)

# # timesteps = start_timesteps.expand(10,-1)
# # print(timesteps)

# # temb = SinusoidalPosEmb(128)

# # output = temb(timesteps)
# # print(output)

# previous_noise = torch.zeros(8,10)
# for i in range(8):
#     previous_noise[i] = ddpm.get_variance_sequence()
# print(previous_noise)
MODEL_PREFIX_LENGTH = 6
import re
def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[MODEL_PREFIX_LENGTH:]  
            new_state_dict[new_k] = v

    return new_state_dict

from diffusion_policy.model.diffusion.stochastic_conditional_unet1d import StochasticConditionalUnet1D

unet = StochasticConditionalUnet1D(
            input_dim=128,
            local_cond_dim=None,
            global_cond_dim=128,
            diffusion_step_embed_dim=128,
            diffusion_timestep=100,
            down_dims=[512, 1024, 2048],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

from diffusion_policy.model.diffusion.conv1d_components import Conv1dBlock

noise_sqence = torch.zeros([1,5,1,3]) # b t h a


#  def get_variance_sequence(self,output_shape):       
#         # b T horizon action_dim
#         variance_squence = torch.zeros(output_shape)
        
#         for t in range(self.num_inference_steps):
#             # variance_squence[i] = self._get_variance(self.timesteps[i])*variance_noise[i]
#             variance_noise = randn_tensor([variance_squence.shape[0],variance_squence.shape[2],variance_squence.shape[3]])
#             variance_squence[:,t,:,:] = self._get_variance(self.timesteps[i])*variance_noise
#         # print(variance_squence)
#         return variance_squence

for t in range(5):
    # variance_squence[i] = self._get_variance(self.timesteps[i])*variance_noise[i]
    variance_noise = torch.arange(1*1*3).view([1,1,3])
    noise_sqence[:,t,:,:] = t*variance_noise
        # print(variance_squence)

print(noise_sqence)


import hydra

# teacher_path = "/home/clear/stochastic_consistency_policy/diffusion_policy_stanford/useful_chckpoint/DDPM/epoch=0700-test_mean_score=0.920.ckpt" 
# teacher = torch.load(teacher_path)
# # print(list(teacher.keys()))
# # print(list(teacher['state_dicts'].keys()))
# print(list(teacher['state_dicts']['model'].keys()))
# state_dict = state_dict_to_model(teacher)
# cfg = teacher['cfg']

# obs_encoder : MultiImageObsEncoder= hydra.utils.instantiate(cfg.policy.obs_encoder)

# # obs_encoder.load_state_dict(teacher['state_dicts']['model'])

# # print(list(teacher['state_dicts']['model'].keys()))

# model_state_dict = {
#     k[len('obs_encoder.'):]: v for k, v in teacher['state_dicts']['model'].items()
#     if k.startswith('obs_encoder.')
# }
# model.load_state_dict(model_state_dict, strict=False)
# print(list(model_state_dict.keys()))
# obs_encoder.load_state_dict(model_state_dict)


# model_state_dict = {
#     k[len('model.diffusion_step_encoder.'):]: v for k, v in teacher['state_dicts']['model'].items()
#     if k.startswith('model.diffusion_step_encoder.')
# }
# unet.diffusion_step_encoder.load_state_dict(model_state_dict)

# model_state_dict = {
#     k[len('model.up_modules.'):]: v for k, v in teacher['state_dicts']['model'].items()
#     if k.startswith('model.up_modules.')
# }
# unet.up_modules.load_state_dict(model_state_dict)

# print(list(model_state_dict.keys()))