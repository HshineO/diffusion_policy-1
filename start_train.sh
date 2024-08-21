export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=4

python train.py --config-dir=. --config-name=train_diffusion_unet_image_workspace.yaml

