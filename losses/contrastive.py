import torch
import math
import torch.nn as nn
import lightning as L

class ContrastiveLoss(L.LightningModule):
    def __init__(self, init_temp=0.07, max_scale=100.0):
        super(ContrastiveLoss, self).__init__()
        init_logit_scale = math.log(1.0 / init_temp)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.max_scale = max_scale

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scale = self.logit_scale.exp().clamp(max=self.max_scale)
        logits_per_image = scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        labels = torch.arange(len(image_features)).to(logits_per_image.device)
        loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
        return 0.5 * (loss_i + loss_t)
