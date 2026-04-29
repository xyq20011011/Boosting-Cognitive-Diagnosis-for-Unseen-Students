import torch
import torch.nn as nn
import torch.nn.functional as F

class NCD(nn.Module):
    def __init__(self, n_exercise, n_concept):
        super().__init__()
        self.diff_emb = nn.Parameter(torch.rand(1, n_exercise, n_concept))
        self.disc_emb = nn.Parameter(torch.rand(1, n_exercise, 1))
        self.n_exercise = n_exercise
        self.mlp = nn.Sequential(
            nn.Linear(n_concept, 512),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.prednet_full1 = nn.Linear(n_concept, 512)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(512, 256)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(256, 1)

    def forward(self,  theta: torch.Tensor, q_matrix: torch.Tensor):
        stu_theta = theta.unsqueeze(1)  # (bs, 1, n_concept)
        h_diff = torch.sigmoid(self.diff_emb)  # (1, n_exercise, n_concept)
        h_disc = torch.sigmoid(self.disc_emb) * 10  # (1, n_exercise, 1)
        x = h_disc * (stu_theta-h_diff) * q_matrix
        out = self.mlp(x)

        return out.squeeze(-1)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)