#!/bin/bash

#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 100000M
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH -o ./slurm_lp/out%j.txt
#SBATCH -e ./slurm_lp/err%j.txt
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@180
#SBATCH --job-name="sc1-finetune"

spack load cuda/s5o57xp
spack load cudnn
source jobs/conf.sh

CHECKPOINT=""

ulimit -n 65535
CUDA_CACHE_DISABLE=1 $PYTHONPATH finetune.py --architecture "scenario_clip_1" \
    --checkpoint_path $CHECKPOINT --metadata_dir '<path_to_metadata_dir>' \
    --lr 2e-5 \
    --classes_json '<path_to_classes_json>' \
    --mode linear-probing \
    --tasks relation \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs 6
