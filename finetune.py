import warnings
warnings.filterwarnings('ignore')

import os
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from transformers import CLIPTokenizer
from models.pyramid_clip import PyramidCLIP
from models.scenario_clip_0 import ScenarioCLIP0
from models.scenario_clip_1 import ScenarioCLIP1
from data.datamodule import ActionGenomeDataModule
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import random
import torch
torch.set_float32_matmul_precision("high")

seed = 42
random.seed(42)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = ArgumentParser()

parser.add_argument('--architecture', type=str, default='pyramid_clip', help='Architecture of the model to be trained', choices=['pyramid_clip', 'scenario_clip_0', 'scenario_clip_1'])
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--resume_path', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--save_dir', type=str, default=f"/scratch/{os.environ.get('USER')}/Exps/scenario-clip", help='Path to the directory to store checkpoints, logs and any other files')
parser.add_argument('--metadata_dir', type=str, default='./metadata', help='Path to the directory containing metadata JSONs')
parser.add_argument('--embedding_storage_dir', type=str, help='Directory to store embeddings')
parser.add_argument('--classes_json', type=str, help='Path to the classes JSON file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
parser.add_argument('--vision_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for vision')
parser.add_argument('--text_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for text')
parser.add_argument('--tokenizer_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained tokenizer for CLIP')
parser.add_argument('--monitor_device_usage_stats', action='store_true', help='Log device usage statistics at each step and epoch')
parser.add_argument('--mode', type=str, choices=['pretrain', 'zero-shot', 'linear-probing'], help='Mode of training: pretrain, zero-shot, or linear-probing')
parser.add_argument('--tasks', nargs='+', help='List of tasks to run in linear probing or zero-shot mode.')
parser.add_argument('--ema', action='store_true', help='EMA based model')
args = parser.parse_args()

if __name__ == "__main__":
    time_now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    data_module = ActionGenomeDataModule(metadata_dir=args.metadata_dir, batch_size=args.batch_size, num_workers=args.num_workers, finetune=True)

    if args.architecture == 'pyramid_clip':
        model = PyramidCLIP(
            vision_model_name=args.vision_model_name,
            text_model_name=args.text_model_name,
            clip_tokenizer_name=args.tokenizer_name,
            embedding_storage_dir=args.embedding_storage_dir,
            classes_json=args.classes_json,
            mode=args.mode,
            tasks=args.tasks
        )
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    elif args.architecture == 'scenario_clip_0':
        model = ScenarioCLIP0(
            vision_model_name=args.vision_model_name,
            text_model_name=args.text_model_name,
            clip_tokenizer_name=args.tokenizer_name,
            embedding_storage_dir=args.embedding_storage_dir,
            classes_json=args.classes_json,
            mode=args.mode,
            tasks=args.tasks
        )
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    elif args.architecture == 'scenario_clip_1':
        if not args.ema:
            model = ScenarioCLIP1(
                vision_model_name=args.vision_model_name,
                text_model_name=args.text_model_name,
                clip_tokenizer_name=args.tokenizer_name,
                embedding_storage_dir=args.embedding_storage_dir,
                classes_json=args.classes_json,
                mode=args.mode,
                tasks=args.tasks
            )
            if args.checkpoint_path:
                checkpoint = torch.load(args.checkpoint_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            if args.checkpoint_path:
                model = ScenarioCLIP1.load_from_checkpoint(
                    args.checkpoint_path,
                    vision_model_name=args.vision_model_name,
                    text_model_name=args.text_model_name,
                    clip_tokenizer_name=args.tokenizer_name,
                    embedding_storage_dir=args.embedding_storage_dir,
                    classes_json=args.classes_json,
                    mode=args.mode,
                    tasks=args.tasks,
                    strict=False
                )
            else:
                model = ScenarioCLIP1(
                    vision_model_name=args.vision_model_name,
                    text_model_name=args.text_model_name,
                    clip_tokenizer_name=args.tokenizer_name,
                    embedding_storage_dir=args.embedding_storage_dir,
                    classes_json=args.classes_json,
                    mode=args.mode,
                    tasks=args.tasks
                )

    csv_logger = CSVLogger(save_dir=f"{args.save_dir}/finetune/linear-probe", name=f"{args.architecture}/bs_{args.batch_size}_lr_{args.lr}", version=time_now)
    tb_logger = TensorBoardLogger(save_dir=f"{args.save_dir}/finetune/linear-probe", name=f"{args.architecture}/bs_{args.batch_size}_lr_{args.lr}", version=time_now)
    print(f"Logging experiment to: {args.save_dir}/finetune/linear-probe/{args.architecture}/bs_{args.batch_size}_lr_{args.lr}/{time_now}")
    logger=[csv_logger, tb_logger]
    
    log_dir = csv_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="epoch={epoch}-step={step}-val_loss={val_loss:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True,
        )

    callbacks_list = [checkpoint_callback]

    if args.monitor_device_usage_stats:
        gpu_stats = DeviceStatsMonitor()
        callbacks_list.append(gpu_stats)

    trainer = L.Trainer(max_epochs=args.max_epochs, devices=1, accelerator="gpu", callbacks=callbacks_list, logger=logger, gradient_clip_algorithm="norm", gradient_clip_val=1.)
    if args.mode == 'linear-probing':
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_path)
    if args.ema:
        trainer.test(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)
