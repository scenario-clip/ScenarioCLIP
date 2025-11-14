import torch.nn as nn
from .distillation import DistillationLoss

class StudentTeacherLoss(nn.Module):
    def __init__(self):
        super(StudentTeacherLoss, self).__init__()
        self.distillation_loss_fn = DistillationLoss()

    def forward(self, global_visual_embeddings, global_text_embeddings, object_visual_embeddings, object_text_embeddings, relation_visual_embeddings, relation_text_embeddings, sizes):

        visual_global_to_object_distill_losses = []
        visual_global_to_relation_distill_losses = []
        text_object_to_global_distill_losses = []
        text_relation_to_global_distill_losses = []

        frozen_global_visual_embeddings = global_visual_embeddings.clone().detach()
        frozen_global_visual_embeddings.requires_grad = False
        frozen_object_text_embeddings = object_text_embeddings.clone().detach()
        frozen_object_text_embeddings.requires_grad = False
        frozen_relation_text_embeddings = relation_text_embeddings.clone().detach()
        frozen_relation_text_embeddings.requires_grad = False

        for i in range(sizes[0]):
            visual_global_to_object_distill_loss = self.distillation_loss_fn.one_teacher_many_students(frozen_global_visual_embeddings[i], object_visual_embeddings[sum(sizes[1][:i]):sum(sizes[1][:i+1])])
            visual_global_to_object_distill_losses.append(visual_global_to_object_distill_loss)
            visual_global_to_relation_distill_loss = self.distillation_loss_fn.one_teacher_many_students(frozen_global_visual_embeddings[i], relation_visual_embeddings[sum(sizes[2][:i]):sum(sizes[2][:i+1])])
            visual_global_to_relation_distill_losses.append(visual_global_to_relation_distill_loss)
            text_object_to_global_distill_loss = self.distillation_loss_fn.one_student_many_teachers(global_text_embeddings[i], frozen_object_text_embeddings[sum(sizes[1][:i]):sum(sizes[1][:i+1])])
            text_object_to_global_distill_losses.append(text_object_to_global_distill_loss)
            text_relation_to_global_distill_loss = self.distillation_loss_fn.one_student_many_teachers(global_text_embeddings[i], frozen_relation_text_embeddings[sum(sizes[2][:i]):sum(sizes[2][:i+1])])
            text_relation_to_global_distill_losses.append(text_relation_to_global_distill_loss)

        visual_global_to_object_distill_loss_mean = sum(visual_global_to_object_distill_losses) / len(visual_global_to_object_distill_losses)
        visual_global_to_relation_distill_loss_mean = sum(visual_global_to_relation_distill_losses) / len(visual_global_to_relation_distill_losses)
        text_object_to_global_distill_loss_mean = sum(text_object_to_global_distill_losses) / len(text_object_to_global_distill_losses)
        text_relation_to_global_distill_loss_mean = sum(text_relation_to_global_distill_losses) / len(text_relation_to_global_distill_losses)

        # total_distill_loss = visual_global_to_object_distill_loss_mean + visual_global_to_relation_distill_loss_mean + text_object_to_global_distill_loss_mean + text_relation_to_global_distill_loss_mean
        distill_object = visual_global_to_object_distill_loss_mean + text_object_to_global_distill_loss_mean
        distill_relation = visual_global_to_relation_distill_loss_mean + text_relation_to_global_distill_loss_mean
        total_distill_loss = distill_object + distill_relation
        return total_distill_loss

