# Scenario CLIP Model

## Environment

Create the environment using [`requirements.txt`](./requirements.txt) or [`env.yml`](./env.yml). Alternatively, create it using the command:

```
conda create -n scenario-clip lightning torchvision pillow transformers -c conda-forge -y
```

Activate the environment:

```
conda activate scenario-clip
```

## Pretraining

### Command

```bash
python pretrain.py --architecture $ARCHITECTURE_NAME \ # pyramid_clip,scenario_clip_{0,1}
    --metadata_dir '<path_to_metadata_dir>' \
    --batch_size 8 --max_epochs 10 --lr 1e-5
```

### Slurm Jobs

Submit jobs for pretrain using the following commands:

```
sbatch jobs/pyramid_clip/pretrain.sh
```

```
sbatch jobs/scenario_clip_0/pretrain.sh
```

```
sbatch jobs/scenario_clip_1/pretrain.sh
```

```
sbatch jobs/scenario_clip_2/pretrain.sh
```
