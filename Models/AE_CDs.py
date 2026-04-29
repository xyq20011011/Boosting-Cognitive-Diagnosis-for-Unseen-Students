import torch
import torch.nn as nn

from Models.Encoders import MlpEncoder, EmbEncoder, MlpEncoder_small, AttentionEncoder, NaiveEncoder
from Models.NCD import NCD


class Emb_NCD(nn.Module):
    def __init__(self, constants, q_matrix):
        super().__init__()
        n_stu, n_exercise, n_concept = constants
        self.assess_net = EmbEncoder(n_stu, n_concept)
        self.CD_model = NCD(n_exercise, n_concept)
        self.Q_matrix = q_matrix

        # initialization
        for name, param in self.named_parameters():
            nn.init.xavier_normal_(param)


    def forward(self, p_matrix, stu_id):
        theta = self.assess_net(stu_id)
        predict = self.CD_model(theta, self.Q_matrix)
        return predict, theta


class VAE_NCD(nn.Module):
    def __init__(self, constants, q_matrix, max_std=0.1):
        super().__init__()
        n_exercise, n_concept = constants[1], constants[2]
        self.assess_net = AttentionEncoder(n_exercise, 2 * n_concept, n_concept=n_concept, out_sigmoid=False, pooling="mean", block_num=1, q_matrix=q_matrix, embedding_dim=32)
        self.CD_model = NCD(n_exercise, n_concept)
        self.Q_matrix = q_matrix
        self.max_std = max_std

        for name, param in self.CD_model.named_parameters():
            if ('weight' in name) or ("emb" in name):
                nn.init.xavier_normal_(param)

    def reparameterize(self, mu, var):
        mu = torch.sigmoid(mu)
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(mu)

        sampled_theta = mu + eps * std
        return sampled_theta

    def forward(self, p_matrix, stu_id):
        theta = self.assess_net(p_matrix)
        mu, var = theta.chunk(2, dim=1)
        if self.training:
            reparameterized_theta = self.reparameterize(mu, var)
        else:
            reparameterized_theta = torch.sigmoid(mu)
        predict = self.CD_model(reparameterized_theta, self.Q_matrix)
        return predict, theta


class AE_NCD(nn.Module):
    def __init__(self, constants, q_matrix):
        super().__init__()
        n_exercise, n_concept = constants[1], constants[2]
        self.assess_net = AttentionEncoder(n_exercise, n_concept, n_concept=n_concept, out_sigmoid=False, pooling="mean", block_num=1, q_matrix=q_matrix, embedding_dim=32)
        self.CD_model = NCD(n_exercise, n_concept)
        self.Q_matrix = q_matrix

    def forward(self, p_matrix, stu_id):
        theta = self.assess_net(p_matrix)
        predict = self.CD_model(theta, self.Q_matrix)
        return predict, theta
