import os
import json
import torch
import functools
import torch.nn as nn
import lightning as L
import torchmetrics as T
from torch.optim import AdamW
from transformers import CLIPTokenizer
from losses.contrastive import ContrastiveLoss
from .backbone import ThreeLevelCLIP
from typing import Optional, List, Literal
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math

class ScenarioCLIP0(L.LightningModule):
    def __init__(self, vision_model_name, text_model_name, clip_tokenizer_name, lr=1e-5, embedding_storage_dir=None, classes_json=None, mode: Literal["pretrain", "zero-shot", "linear-probing"] = "pretrain", tasks: Optional[List[str]] = None):
        super().__init__()
        self.model = ThreeLevelCLIP(vision_model_name, text_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_name)
        # self.contrastive_loss_fn = ContrastiveLoss()
        self.global_loss = ContrastiveLoss()
        self.object_loss = ContrastiveLoss()
        self.relation_loss = ContrastiveLoss()
        self.lr = lr
        self.mode = mode
        if self.mode == "zero-shot":
            self.embedding_storage_dir = embedding_storage_dir
            self.classes_json = classes_json

            # Zero-shot retrieval
            self.action_test = "action" in tasks if tasks else True
            self.object_test = "object" in tasks if tasks else True
            self.relation_test = "relation" in tasks if tasks else True
            if self.action_test:
                self.actions = self.action_classlist()
                self.action_embeddings = []
                for k in [1, 5, 10]:
                    setattr(self, f"correct_action_top{k}", 0)
                    setattr(self, f"incorrect_action_top{k}", 0)
            if self.object_test:
                self.objects = self.object_classlist()
                self.object_embeddings = []
                for k in [1, 5, 10]:
                    setattr(self, f"correct_object_top{k}", 0)
                    setattr(self, f"incorrect_object_top{k}", 0)
            if self.relation_test:
                self.relations = self.relation_classlist()
                self.relation_embeddings = []
                for k in [1, 5, 10]:
                    setattr(self, f"correct_relation_top{k}", 0)
                    setattr(self, f"incorrect_relation_top{k}", 0)
        if self.mode == "linear-probing":
            print(f"Tasks is {tasks}")
            self.classes_json = classes_json
            for param in self.model.parameters():
                param.requires_grad = False
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            if "action" in tasks:
                self.action_test = True
                self.object_test = False
                self.relation_test = False
                for k in [1, 5, 10]:
                    setattr(self, f"correct_action_top{k}", 0)
                    setattr(self, f"incorrect_action_top{k}", 0)
                self.actions = self.action_classlist()
                self.action_head = nn.Linear(512, len(self.actions))
            elif "object" in tasks:
                self.action_test = False
                self.object_test = True
                self.relation_test = False
                for k in [1, 5, 10]:
                    setattr(self, f"correct_object_top{k}", 0)
                    setattr(self, f"incorrect_object_top{k}", 0)
                self.objects = self.object_classlist()
                self.object_head = nn.Linear(512, len(self.objects))
            elif "relation" in tasks:
                self.action_test = False
                self.object_test = False
                self.relation_test = True
                for k in [1, 5, 10]:
                    setattr(self, f"correct_relation_top{k}", 0)
                    setattr(self, f"incorrect_relation_top{k}", 0)
                self.relations = self.relation_classlist()
                self.relation_head = nn.Linear(512, len(self.relations))
    
    def configure_optimizers(self, warmup_ratio: float = 0.10, weight_decay: float = 0.2, lr_floor: float = 0.0):
        if self.mode == "linear-probing":
            trainable = [p for p in self.parameters() if p.requires_grad]
            opt = AdamW(trainable, lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
            return opt

        def no_decay(name: str) -> bool:
            return name.endswith(".bias") or ("norm" in name.lower()) or ("ln" in name.lower())

        named = list(self.model.named_parameters())
        named += [(f"global_loss.{n}", p)   for n, p in self.global_loss.named_parameters()]
        named += [(f"object_loss.{n}", p)   for n, p in self.object_loss.named_parameters()]
        named += [(f"relation_loss.{n}", p) for n, p in self.relation_loss.named_parameters()]

        decay, nodecay = [], []
        for n, p in named:
            if p.requires_grad:
                (nodecay if no_decay(n) else decay).append(p)

        opt = AdamW(
            [{"params": decay,   "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}],
            lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )

        # total training steps
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = steps_per_epoch * self.trainer.max_epochs

        warmup_steps  = max(1, int(warmup_ratio * total_steps))
        cosine_steps  = max(1, total_steps - warmup_steps)

        sched_warm = LinearLR(opt, start_factor=1e-3, total_iters=warmup_steps)
        sched_cos  = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=lr_floor)

        sched = SequentialLR(opt, schedulers=[sched_warm, sched_cos], milestones=[warmup_steps])

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }


    def training_step(self, batch, batch_idx):
        # Global Level

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch
        if self.mode == 'pretrain':

            caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            global_visual_embeddings = self.model.global_encode_image(images)

            global_text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

            # Object Level

            object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
            object_captions = [caption for cap_list in object_names for caption in cap_list]
            object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            object_visual_embeddings = self.model.object_encode_image(object_images)
            object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

            # Relation Level

            relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
            relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
            relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

            relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
            relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

            global_contrastive_loss = self.global_loss(global_visual_embeddings, global_text_embeddings)
            object_contrastive_loss = self.object_loss(object_visual_embeddings, object_text_embeddings)
            relation_contrastive_loss = self.relation_loss(relation_visual_embeddings, relation_text_embeddings)

            contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

            self.log_dict({
                'train_contrastive_loss': contrastive_loss
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return contrastive_loss
        elif self.mode == "linear-probing":
            loss = 0.

            if self.action_test:
                labels = torch.tensor([self.actions.index(caption) for caption in captions]).to(self.device)
                with torch.no_grad():
                    global_visual_embeddings = self.model.global_encode_image(images)
                logits = self.action_head(global_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)

            elif self.object_test:
                object_captions = [caption for cap_list in object_names for caption in cap_list]
                labels = torch.tensor([self.objects.index(caption) for caption in object_captions]).to(self.device)
                object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
                with torch.no_grad():
                    object_visual_embeddings = self.model.object_encode_image(object_images)
                logits = self.object_head(object_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)

            elif self.relation_test:
                relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
                labels = torch.tensor([self.relations.index(caption) for caption in relation_captions_flattened]).to(self.device)
                relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
                with torch.no_grad():
                    relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
                logits = self.relation_head(relation_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
            return loss

    def validation_step(self, batch, batch_idx):

        # Global Level

        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch

        if self.mode == 'pretrain':
            caption_tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            global_visual_embeddings = self.model.global_encode_image(images)

            global_text_embeddings = self.model.global_encode_text(caption_tokens.input_ids)

            # Object Level

            object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
            object_captions = [caption for cap_list in object_names for caption in cap_list]
            object_tokens = self.tokenizer(object_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

            object_visual_embeddings = self.model.object_encode_image(object_images)
            object_text_embeddings = self.model.object_encode_text(object_tokens.input_ids)

            # Relation Level

            relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
            relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
            relation_tokens = self.tokenizer(relation_captions_flattened, return_tensors="pt", padding=True, truncation=True).to(self.device)

            relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
            relation_text_embeddings = self.model.relation_encode_text(relation_tokens.input_ids)

            # Loss Calculation

            global_contrastive_loss = self.global_loss(global_visual_embeddings, global_text_embeddings)
            object_contrastive_loss = self.object_loss(object_visual_embeddings, object_text_embeddings)
            relation_contrastive_loss = self.relation_loss(relation_visual_embeddings, relation_text_embeddings)

            contrastive_loss = global_contrastive_loss + object_contrastive_loss + relation_contrastive_loss

            self.log_dict({
                'val_contrastive_loss': contrastive_loss
            }, prog_bar=False, logger=True, on_epoch=True, on_step=True)

            return contrastive_loss
        elif self.mode == "linear-probing":
            loss = 0.

            if self.action_test:
                labels = torch.tensor([self.actions.index(caption) for caption in captions]).to(self.device)
                with torch.no_grad():
                    global_visual_embeddings = self.model.global_encode_image(images)
                logits = self.action_head(global_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean()
                self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
                self.log('val_acc', acc, prog_bar=False, logger=True, on_epoch=True, on_step=True)
            elif self.object_test:
                object_captions = [caption for cap_list in object_names for caption in cap_list]
                labels = torch.tensor([self.objects.index(caption) for caption in object_captions]).to(self.device)
                object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
                with torch.no_grad():
                    object_visual_embeddings = self.model.object_encode_image(object_images)
                logits = self.object_head(object_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean()
                self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
                self.log('val_acc', acc, prog_bar=False, logger=True, on_epoch=True, on_step=True)
            elif self.relation_test:
                relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
                labels = torch.tensor([self.relations.index(caption) for caption in relation_captions_flattened]).to(self.device)
                relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
                with torch.no_grad():
                    relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
                logits = self.relation_head(relation_visual_embeddings)
                loss = self.cross_entropy_loss(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == labels).float().mean()
                self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
                self.log('val_acc', acc, prog_bar=False, logger=True, on_epoch=True, on_step=True)
            return loss

    @functools.cache
    def action_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['actions']

    @functools.cache
    def object_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['objects']

    @functools.cache
    def relation_classlist(self):
        with open(self.classes_json) as f:
            classes = json.load(f)
        return classes['relations']

    def on_test_start(self):

        # Store action embeddings
        if self.mode == "zero-shot":
            if self.action_test:
                action_dir = f"{self.embedding_storage_dir}/actions"
                os.makedirs(action_dir, exist_ok=True)
                if len(os.listdir(action_dir)) > 0:
                    print("Action embeddings already exist. Skipping storage.")
                    for i in range(len(self.actions)):
                        embedding = torch.load(f"{action_dir}/action_{i}.pt", map_location=self.device)
                        self.action_embeddings.append(embedding.squeeze(0))
                else:
                    for i, action in enumerate(self.actions):
                        caption_tokens = self.tokenizer(action, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        action_embedding = self.model.global_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                        self.action_embeddings.append(action_embedding.squeeze(0))
                        torch.save(action_embedding, f"{action_dir}/action_{i}.pt")
                self.correct_action_predictions = 0
                self.incorrect_action_predictions = 0
                self.action_embeddings = torch.stack(self.action_embeddings)
                self.action_embeddings = torch.nn.functional.normalize(self.action_embeddings, dim=-1)
                print("Saved Action Embeddings.")


            # Store object embeddings

            if self.object_test:
                object_dir = f"{self.embedding_storage_dir}/objects"
                os.makedirs(object_dir, exist_ok=True)
                if len(os.listdir(object_dir)) > 0:
                    print("Object embeddings already exist. Skipping storage.")
                    for i in range(len(self.objects)):
                        embedding = torch.load(f"{object_dir}/object_{i}.pt", map_location=self.device)
                        self.object_embeddings.append(embedding.squeeze(0))
                else:
                    for i, object in enumerate(self.objects):
                        caption_tokens = self.tokenizer(object, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        object_embedding = self.model.object_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                        self.object_embeddings.append(object_embedding.squeeze(0))
                        torch.save(object_embedding, f"{self.embedding_storage_dir}/objects/object_{i}.pt")

                self.correct_object_predictions = 0
                self.incorrect_object_predictions = 0
                self.object_embeddings = torch.stack(self.object_embeddings)
                self.object_embeddings = torch.nn.functional.normalize(self.object_embeddings, dim=-1)
                print("Saved Object Embeddings.")

            # Store relation embeddings

            if self.relation_test:
                relation_dir = f"{self.embedding_storage_dir}/relations"
                os.makedirs(relation_dir, exist_ok=True)
                if len(os.listdir(relation_dir)) > 0:
                    print("Relation embeddings already exist. Skipping storage.")
                    for i in range(len(self.relations)):
                        embedding = torch.load(f"{relation_dir}/relation_{i}.pt", map_location=self.device)
                        self.relation_embeddings.append(embedding.squeeze(0))
                else:
                    for i, relation in enumerate(self.relations):
                        caption_tokens = self.tokenizer(relation, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        relation_embedding = self.model.relation_encode_text(torch.unsqueeze(caption_tokens.input_ids, 0))
                        self.relation_embeddings.append(relation_embedding.squeeze(0))
                        torch.save(relation_embedding, f"{self.embedding_storage_dir}/relations/relation_{i}.pt")

                self.correct_relation_predictions = 0
                self.incorrect_relation_predictions = 0
                self.relation_embeddings = torch.stack(self.relation_embeddings)
                self.relation_embeddings = torch.nn.functional.normalize(self.relation_embeddings, dim=-1)
                print("Saved Relation Embeddings.")


    def test_step(self, batch, batch_idx):
        images, captions, object_names, objects_cropped, relation_captions, relation_images = batch
        def increment(attr):
            setattr(self, attr, getattr(self, attr) + 1)
        if self.mode == "zero-shot":
            if self.action_test:
                image_embeddings = self.model.global_encode_image(images)
                image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
                similarities = image_embeddings @ self.action_embeddings.T  # shape: [num_images, num_actions]

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(similarities, k, dim=-1).indices for k in topk}  # [num_images, k]

                for i in range(len(images)):
                    for k in topk:
                        topk_indices = topk_preds[k][i].tolist()
                        topk_labels = [self.actions[j] for j in topk_indices]

                        if captions[i] in topk_labels:
                            increment(f"correct_action_top{k}")
                        else:
                            increment(f"incorrect_action_top{k}")

            # Object Recognition
            if self.object_test:
                object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
                object_embeddings = self.model.object_encode_image(object_images)

                norm_embeddings = torch.nn.functional.normalize(object_embeddings, dim=-1)
                similarities = norm_embeddings @ self.object_embeddings.T

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(similarities, k, dim=-1).indices for k in topk}

                offset = 0
                for true_obj_list in object_names:
                    num_objects = len(true_obj_list)
                    for i in range(num_objects):
                        idx = offset + i
                        for k in topk:
                            topk_indices = topk_preds[k][idx].tolist()
                            topk_labels = [self.objects[j] for j in topk_indices]

                            if any(pred in true_obj_list for pred in topk_labels):
                                increment(f"correct_object_top{k}")
                            else:
                                increment(f"incorrect_object_top{k}")
                    offset += num_objects

            # Relation Recognition
            if self.relation_test:
                relation_images_flat = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
                relation_embeddings = self.model.relation_encode_image(relation_images_flat)

                norm_embeddings = torch.nn.functional.normalize(relation_embeddings, dim=-1)
                similarities = norm_embeddings @ self.relation_embeddings.T  # shape: [num_preds, num_classes]

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(similarities, k, dim=-1).indices for k in topk}

                offset = 0
                for caption_token_lists in relation_captions:
                    num_relations = len(caption_token_lists)
                    true_rel_strings = [' '.join(tokens) for tokens in caption_token_lists]

                    for i in range(num_relations):
                        idx = offset + i
                        for k in topk:
                            topk_indices = topk_preds[k][idx].tolist()
                            topk_labels = [self.relations[j] for j in topk_indices]

                            if any(pred in true_rel_strings for pred in topk_labels):
                                increment(f"correct_relation_top{k}")
                            else:
                                increment(f"incorrect_relation_top{k}")
                    offset += num_relations
        elif self.mode == "linear-probing":
            if self.action_test:
                labels = torch.tensor([self.actions.index(caption) for caption in captions]).to(self.device)
                global_visual_embeddings = self.model.global_encode_image(images)
                logits = self.action_head(global_visual_embeddings)

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(logits, k, dim=-1).indices for k in topk}

                for i, label in enumerate(labels):
                    for k in topk:
                        topk_indices = topk_preds[k][i].tolist()
                        topk_labels = [self.actions[j] for j in topk_indices]

                        if self.actions[label.item()] in topk_labels:
                            increment(f"correct_action_top{k}")
                        else:
                            increment(f"incorrect_action_top{k}")

            elif self.object_test:
                object_captions = [caption for cap_list in object_names for caption in cap_list]
                labels = torch.tensor([self.objects.index(caption) for caption in object_captions]).to(self.device)
                object_images = torch.stack([img for obj_list in objects_cropped for img in obj_list]).to(self.device)
                object_visual_embeddings = self.model.object_encode_image(object_images)
                logits = self.object_head(object_visual_embeddings)

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(logits, k, dim=-1).indices for k in topk}

                offset = 0
                for true_obj_list in object_names:
                    num_objects = len(true_obj_list)
                    for i in range(num_objects):
                        idx = offset + i
                        for k in topk:
                            topk_indices = topk_preds[k][idx].tolist()
                            topk_labels = [self.objects[j] for j in topk_indices]

                            if any(pred in true_obj_list for pred in topk_labels):
                                increment(f"correct_object_top{k}")
                            else:
                                increment(f"incorrect_object_top{k}")
                    offset += num_objects

            elif self.relation_test:
                relation_captions_flattened = [' '.join(caption_list) for cap_list in relation_captions for caption_list in cap_list]
                labels = torch.tensor([self.relations.index(caption) for caption in relation_captions_flattened]).to(self.device)
                relation_images_stacked = torch.stack([img for rel_list in relation_images for img in rel_list]).to(self.device)
                relation_visual_embeddings = self.model.relation_encode_image(relation_images_stacked)
                logits = self.relation_head(relation_visual_embeddings)

                topk = [1, 5, 10]
                topk_preds = {k: torch.topk(logits, k, dim=-1).indices for k in topk}

                offset = 0
                for caption_token_lists in relation_captions:
                    num_relations = len(caption_token_lists)
                    true_rel_strings = [' '.join(tokens) for tokens in caption_token_lists]

                    for i in range(num_relations):
                        idx = offset + i
                        for k in topk:
                            topk_indices = topk_preds[k][idx].tolist()
                            topk_labels = [self.relations[j] for j in topk_indices]

                            if any(pred in true_rel_strings for pred in topk_labels):
                                increment(f"correct_relation_top{k}")
                            else:
                                increment(f"incorrect_relation_top{k}")
                    offset += num_relations

    def on_test_end(self):
        if self.mode == "zero-shot":
            if self.action_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_action_top{k}")
                    incorrect = getattr(self, f"incorrect_action_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Action Top-{k} Predictions: {correct} Correct, {incorrect} Incorrect")
                    print(f"Action Recognition Top-{k} Accuracy = {acc:.4f}")

            if self.object_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_object_top{k}")
                    incorrect = getattr(self, f"incorrect_object_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Object Top-{k} Predictions: {correct} Correct, {incorrect} Incorrect")
                    print(f"Object Recognition Top-{k} Accuracy = {acc:.4f}")

            if self.relation_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_relation_top{k}")
                    incorrect = getattr(self, f"incorrect_relation_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Relation Top-{k} Predictions: {correct} Correct, {incorrect} Incorrect")
                    print(f"Relation Recognition Top-{k} Accuracy = {acc:.4f}")

        elif self.mode == "linear-probing":
            if self.action_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_action_top{k}")
                    incorrect = getattr(self, f"incorrect_action_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Action Recognition Top-{k} Accuracy = {acc:.4f}")

            if self.object_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_object_top{k}")
                    incorrect = getattr(self, f"incorrect_object_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Object Recognition Top-{k} Accuracy = {acc:.4f}")

            if self.relation_test:
                for k in [1, 5, 10]:
                    correct = getattr(self, f"correct_relation_top{k}")
                    incorrect = getattr(self, f"incorrect_relation_top{k}")
                    total = correct + incorrect
                    acc = correct / total if total > 0 else 0.0
                    print(f"Relation Recognition Top-{k} Accuracy = {acc:.4f}")
