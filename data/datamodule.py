import torch
import lightning as L
from torch.utils.data import DataLoader
from .dataset import ActionGenomeDataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as I
import random
import json
import glob

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class ActionGenomeDataModule(L.LightningDataModule):
    def __init__(self, metadata_dir, batch_size=8, num_workers=16, finetune=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.finetune = finetune
        self.global_train_preprocess = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=I.BICUBIC),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.10),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

        self.obj_train_preprocess = T.Compose([
            T.Resize(224, interpolation=I.BICUBIC),
            T.CenterCrop(224),
            T.RandomApply([T.ColorJitter(0.15, 0.15, 0.15, 0.03)], p=0.2),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

        self.rel_train_preprocess = T.Compose([
            T.Resize(224, interpolation=I.BICUBIC),
            T.CenterCrop(224),
            T.RandomApply([T.ColorJitter(0.10, 0.10, 0.10, 0.02)], p=0.2),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

        self.val_preprocess = T.Compose([
            T.Resize(224, interpolation=I.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self.metadata_dir = metadata_dir
        self.metadata = []
        for file in sorted(glob.glob(f"{metadata_dir}/metadata*.json")):
            with open(file) as f:
                self.metadata.extend(json.load(f)['data'])
        with open(f"{metadata_dir}/classes.json") as f:
            self.classes = json.load(f)

    def worker_init_fn(self, worker_id):
        info = torch.utils.data.get_worker_info()
        worker_seed = info.seed
        info.dataset.worker_base_seed = worker_seed
    
    def collate(self, batch):
        image, caption, object_names, objects_cropped, relation_captions, relation_images = zip(*batch)
        images = torch.stack(image)
        return images, caption, object_names, objects_cropped, relation_captions, relation_images

    def setup(self, stage=None):
        random.shuffle(self.metadata)
        train_count = int(len(self.metadata)*.7)
        test_count = int(len(self.metadata)*.2)
        self.train_metadata = self.metadata[:train_count]
        self.val_metadata = self.metadata[train_count:-test_count]
        self.test_metadata = self.metadata[-test_count:]
        self.train_dataset = ActionGenomeDataset(global_transform=self.global_train_preprocess, obj_transform=self.obj_train_preprocess, rel_transform=self.rel_train_preprocess, metadata=self.train_metadata)
        self.val_dataset = ActionGenomeDataset(global_transform=self.val_preprocess, obj_transform=self.val_preprocess, rel_transform=self.val_preprocess, metadata=self.val_metadata)
        self.test_dataset = ActionGenomeDataset(global_transform=self.val_preprocess, obj_transform=self.val_preprocess, rel_transform=self.val_preprocess, metadata=self.test_metadata)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True, worker_init_fn=self.worker_init_fn) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True, worker_init_fn=self.worker_init_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True, worker_init_fn=self.worker_init_fn)
