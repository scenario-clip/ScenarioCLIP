import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class DistillationLoss(L.LightningModule):
    def __init__(self, T_teacher=0.07, T_student=0.07, learnable=True):
        super(DistillationLoss, self).__init__()
        self.T_teacher = T_teacher
        if learnable:
            init = torch.log(torch.tensor(1.0 / T_student))
            self.logit_scale_s = nn.Parameter(init)
        else:
            self.register_buffer("logit_scale_s", torch.log(torch.tensor(1.0 / T_student)))
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # KL divergence loss

    def _T_s(self):
        s = self.logit_scale_s.exp().clamp(min=1.0, max=100.0)
        return 1.0 / s

    def one_teacher_many_students(self, teacher_logits, student_group):
        T_t = self.T_teacher
        T_s = self._T_s()
        teacher_probs = F.softmax(teacher_logits / T_t, dim=-1)
        teacher_probs = teacher_probs.clamp(min=1e-8, max=1.0)

        distill_loss = 0.
        for i in range(len(student_group)):
            student_log_probs = F.log_softmax(student_group[i] / T_s, dim=-1)
            student_log_probs = student_log_probs
            distill_loss += self.kl_div_loss(student_log_probs, teacher_probs) * (T_s ** 2)

        distill_loss = distill_loss / len(student_group)

        return distill_loss
    
    def one_student_many_teachers(self, student_logits, teacher_group):
        T_t = self.T_teacher
        T_s = self._T_s()
        student_probs = F.log_softmax(student_logits / T_s, dim=-1)
        student_probs = student_probs
        distill_loss = 0.
        for i in range(len(teacher_group)):
            teacher_log_probs = F.softmax(teacher_group[i] / T_t, dim=-1)
            teacher_log_probs = teacher_log_probs.clamp(min=1e-8, max=1.0)
            distill_loss += self.kl_div_loss(student_probs, teacher_log_probs) * (T_s ** 2)
        distill_loss = distill_loss / len(teacher_group)
        return distill_loss

