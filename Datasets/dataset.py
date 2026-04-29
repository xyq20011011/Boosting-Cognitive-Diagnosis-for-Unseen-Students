import json
import torch
import os
import copy
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, SequentialSampler
import numpy as np
from Datasets.Junyi import Junyi
import config


class CdDataset(Dataset):
    def __init__(self, data_pth, q_pth, n_exercise):
        with open(data_pth, encoding='utf8') as i_f:
            log_dict = json.load(i_f)
            self.data = list(log_dict.items())
        self.Q_matrix = torch.load(q_pth)
        self.n_exercise = n_exercise
        self.skip_percent = config.stu_mask_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        response_dict = self.data[item][1]
        keys = list(response_dict.keys())
        num_keys = len(keys)
        num_to_skip = int(num_keys * self.skip_percent)
        np.random.seed(item)
        keys_to_skip = set(np.random.choice(keys, num_to_skip, replace=False))  # 需要被掩盖的题目

        P_matrix = torch.zeros(self.n_exercise)
        target_P_matrix = torch.zeros(self.n_exercise)

        for p_id, response in self.data[item][1].items():
            if p_id not in keys_to_skip:
                P_matrix[int(p_id)] = response + 1
            target_P_matrix[int(p_id)] = response + 1

        return P_matrix, target_P_matrix, item


def generate_dataloader(full_dataset, batch_size=32, shuffle=True):

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)


    test_size = int(dataset_size * config.stu_test_rate)


    test_start = 0
    test_end = test_size
    test_indices = indices[test_start:test_end]


    remaining_indices = indices[:test_start] + indices[test_end:]
    train_size = int(0.8 * len(remaining_indices))
    train_indices = remaining_indices[:train_size]
    val_indices = remaining_indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    return train_loader, val_loader, test_loader


def get_dataloader(name="Junyi", batch_size=32, fold=1):
    dataset = None
    constant = None

    if name == "Junyi":
        dataset = CdDataset("./Datasets/Junyi/log.json",
                            "./Datasets/Junyi/Q-matrix.pt", Junyi.N_EXERCISE)
        constant = (Junyi.N_STU, Junyi.N_EXERCISE, Junyi.N_CONCEPT)

    train_loader, val_loader, test_loader = generate_dataloader(dataset, batch_size=batch_size)
    q_matrix = dataset.Q_matrix
    return train_loader, val_loader, test_loader, q_matrix, constant


