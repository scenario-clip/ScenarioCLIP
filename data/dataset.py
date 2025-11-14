from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from .utils import temp_seed
import hashlib

class ActionGenomeDataset(Dataset):
    def __init__(self, metadata, global_transform=None, obj_transform=None, rel_transform=None):
        self.global_transform = global_transform
        self.obj_transform = obj_transform
        self.rel_transform = rel_transform
        self.metadata = metadata

    def _sample_seed(self, idx):
        base = getattr(self, "worker_base_seed", 0)  # set in worker_init_fn
        key = f"{self.metadata[idx]['image_path']}::{idx}::{base}".encode()
        h = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(h, "little") & 0x7fffffff
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata[idx]
        try:
            if ('mnt/MIG_store/Datasets/' in data['image_path']):
                new_path = data['image_path']
                img = Image.open(new_path).convert("RGB")
            else:
                new_path = data['image_path']
                img = Image.open(new_path).convert("RGB")
        except Exception as e:
            img1 = Image.new("RGB", (256, 256), color="white")
        s = self._sample_seed(idx)
        with temp_seed(s): image = self.global_transform(img)
        caption = data['action']
        object_names = [x[0] for x in data['objects']]
        bboxes = [x[1] for x in data['objects']]
        objects_cropped = []
        for bbox in bboxes:
            bbox = [int(b) for b in bbox]
            img1 = Image.new(img.mode, img.size, color='black')
            img1.paste(img.crop(bbox), bbox)
            with temp_seed(s): img1 = self.obj_transform(img1)
            objects_cropped.append(img1)
        relation_images_list = [x[0] for x in data['relations']]
        relation_captions = [x[1] for x in data['relations']]
        relation_images = []
        for focused_region in relation_images_list:
            try: img1 = Image.open(focused_region).convert("RGB")
            except: img1 = Image.new("RGB", (256, 256), color="white")
            with temp_seed(s): img1 = self.rel_transform(img1)
            relation_images.append(img1)
        return image, caption, object_names, objects_cropped, relation_captions, relation_images
