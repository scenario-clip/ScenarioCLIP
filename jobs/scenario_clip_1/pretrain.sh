#!/bin/bash

#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o ./slurm/out%j.txt
#SBATCH -e ./slurm/err%j.txt
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@180
#SBATCH --job-name="sc1-pretrain"

spack load cuda/s5o57xp
spack load cudnn
source jobs/conf.sh

ulimit -n 65535
CUDA_CACHE_DISABLE=1 $PYTHONPATH pretrain.py --architecture "scenario_clip_1" \
    --metadata_dir '<path_to_metadata_dir>' \
    --batch_size 128 --max_epochs 12 --lr 2e-5 --num_workers 8 \
