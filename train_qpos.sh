#!/bin/bash

task_name=(insertion)
steps=(100)
max_train_steps=200000

for step in ${steps[@]}
do
    
    if [ -d "outputs/train/diffusion_transformer_${task_name}_step${step}_use_ss/checkpoints/last" ]; then
        if [ -d "outputs/train/diffusion_tramsformer_${task_name}_step${step}_use_ss/checkpoints/0${max_train_steps}" ]; then
            continue
        else
            python lerobot/scripts/train.py hydra.run.dir=outputs/train/diffusion_transformer_${task_name}_step${step}_use_ss resume=true
        fi
    else
        python  lerobot/scripts/train.py --config-dir lerobot/configs/policy --config-name diffusion_transformer_aloha dataset_repo_id=aloha_sim_${task_name}_scripted_50 env.task=AlohaInsertion-v0 training.batch_size=8 policy.num_train_timesteps=${step} hydra.run.dir=outputs/train/diffusion_transformer_${task_name}_step${step}_use_ss
    fi
done