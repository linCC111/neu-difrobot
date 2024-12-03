#!/bin/bash

steps=(10 50 100 150)
max_train_steps=100000

for step in ${steps[@]}
do
    
    if [ -d "outputs/train/diffusion_transformer_transfer_cube_step${step}/checkpoints/last" ]; then
        if [ -d "outputs/train/diffusion_transformer_transfer_cube_step${step}/checkpoints/0${max_train_steps}" ]; then
            continue
        else
            python lerobot/scripts/train.py hydra.run.dir=outputs/train/diffusion_transformer_transfer_cube_step${step} resume=true
        fi
    else
        python  lerobot/scripts/train.py --config-dir lerobot/configs/policy --config-name diffusion_transformer_aloha training.offline_steps=${max_train_steps} policy.num_train_timesteps=${step} policy.num_inference_steps=${step} hydra.run.dir=outputs/train/diffusion_transformer_transfer_cube_step${step}
    fi
done