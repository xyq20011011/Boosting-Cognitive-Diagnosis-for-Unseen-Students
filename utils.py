import torch
import matplotlib.pyplot as plt

def get_model_name(model):
    model_repr = repr(model)
    class_name = model_repr.split("'")[1]
    model_name = class_name.split(".")[2]
    return model_name


def create_random_nonzero_mask(p_matrix):
    num_rows, num_cols = p_matrix.shape
    mask = torch.zeros_like(p_matrix, dtype=torch.bool)
    for col in range(num_cols):
        non_zero_indices = torch.nonzero(p_matrix[:, col] != 0, as_tuple=False)
        if non_zero_indices.numel() == 0:
            continue
        chosen_index = non_zero_indices[torch.randint(0, non_zero_indices.size(0), (1,))].item()
        mask[chosen_index, col] = True
    return mask



def random_mask_target_p(p, rate):
    target_p = p.clone()
    batch_size, n = target_p.shape
    for i in range(batch_size):
        non_zero_indices = (target_p[i] != 0).nonzero(as_tuple=False).squeeze()

        if len(non_zero_indices) > 0:
            num_to_zero = int(len(non_zero_indices) * rate)
            indices_to_zero = torch.randperm(len(non_zero_indices))[:num_to_zero]
            target_p[i, non_zero_indices[indices_to_zero]] = 0
    return target_p


class ModelInfo:
    def __init__(self):
        self.current_epoch = 0
        self.losses = []
        self.val_aucs = []
        self.test_aucs = []
        self.val_accs = []
        self.test_accs = []
        self.val_rmse = []
        self.test_rmse = []

    def add(self, train_loss, val_auc, test_auc, val_acc=None, test_acc=None, val_rmse=None, test_rmse=None):
        self.current_epoch += 1
        self.losses.append(train_loss)
        self.val_aucs.append(val_auc)
        self.test_aucs.append(test_auc)

        self.val_accs.append(val_acc)
        self.test_accs.append(test_acc)

        self.val_rmse.append(val_rmse)
        self.test_rmse.append(test_rmse)

    def is_best(self):
        current_auc = self.val_aucs[-1]
        for auc in self.val_aucs[:-1]:
            if auc > current_auc:
                return False
        return True

    def best(self):
        best_auc = 0
        best_test = None
        best_val_acc = None
        best_test_acc = None
        best_val_rmse = None
        best_test_rmse = None
        for i, auc in enumerate(self.val_aucs):
            if auc > best_auc:
                best_auc = auc
                best_test = self.test_aucs[i]
                best_val_acc = self.val_accs[i]
                best_test_acc = self.test_accs[i]
                best_val_rmse = self.val_rmse[i]
                best_test_rmse = self.test_rmse[i]

        return best_auc, best_test, best_val_acc, best_test_acc, best_val_rmse, best_test_rmse

    def best_target(self):
        best_target_auc = 0
        best_target_acc = 0
        best_target_rmse = 0
        for i, auc in enumerate(self.test_aucs):
            if auc > best_target_auc:
                best_target_auc = auc
                best_target_acc = self.test_accs[i]
                best_target_rmse = self.test_rmse[i]
        return best_target_acc, best_target_auc, best_target_rmse

    def plot(self, title=None):
        epochs = range(self.current_epoch)
        plt.plot(epochs, self.losses, label='Training Loss', marker='o')
        plt.title(f'{title} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(epochs, self.val_aucs, label='Validation AUC', marker='o', color='orange')
        plt.plot(epochs, self.test_aucs, label='Test AUC', marker='^', color='green')
        plt.title(f'{title} Validation and Test AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.show()

    def best_epoch(self):
        best_auc = 0
        best_epoch = -1
        for i, auc in enumerate(self.val_aucs):
            if auc > best_auc:
                best_auc = auc
                best_epoch = i
        return best_epoch
