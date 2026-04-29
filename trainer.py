from Datasets.dataset import get_dataloader
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from tqdm import tqdm
import numpy as np
import seaborn as sns
import time
from utils import *


class Trainer:
    def __init__(self, model_name, data_path="data"):
        self.constant = None
        self.Q_matrix = None
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.dataset_name = None
        self.learning_rate = 5e-4
        self.weight_decay = 0
        self.verbose = True
        self.show_plot = False

        self.model_path = os.path.join("saved_model", model_name)
        self.model_name = model_name
        self.model = None
        self.model_info = None
        self.gpu = True

    def init_model(self, model_classname):
        self.print("Initializing...")
        if not os.path.exists(self.model_path):
            self.print(f"{self.model_path} created")
            os.makedirs(self.model_path)

        self.model = model_classname(self.constant, self.Q_matrix.cuda())
        if self.gpu:
            self.model.cuda()
        self.model_info = ModelInfo()

    def remove(self):
        print("Are you sure to remove ", self.model_name, "?[yes/no]")
        if input() == "yes":
            if os.path.exists(self.model_path):
                file_list = os.listdir(self.model_path)
                for file_name in file_list:
                    file_path = os.path.join(self.model_path, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        else:
            exit()

    def exist_model(self):
        torch_model_pth = os.path.join(self.model_path, "model")
        return os.path.exists(torch_model_pth)

    def load_model(self):
        torch_model_pth = os.path.join(self.model_path, "model")
        assert os.path.exists(torch_model_pth)
        self.print(f"Model loaded from {self.model_path}")
        self.model = torch.load(torch_model_pth)
        if self.gpu:
            self.model.cuda()

        data_pth = os.path.join(self.model_path, "info.pkl")
        with open(data_pth, 'rb') as file:
            self.model_info = pickle.load(file)

    def save_model(self):
        assert self.model is not None
        self.print(f"Model saved to {self.model_path}")
        torch_model_pth = os.path.join(self.model_path, "model")
        torch.save(self.model, torch_model_pth)

        data_pth = os.path.join(self.model_path, "info.pkl")
        with open(data_pth, 'wb') as file:
            pickle.dump(self.model_info, file)

    def load_data(self, name="ASSIST", batch_size=32, fold=1):
        self.dataset_name = name
        self.print(f"Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.Q_matrix, self.constant = get_dataloader(name=name,
                                                                                                            batch_size=batch_size)


    def train(self, num_epoch=None, to_epoch=None, loss_function=None, mask_ratio=0.6, mute_result=False):
        assert self.model is not None
        assert self.train_loader is not None
        if to_epoch is not None:
            num_epoch = to_epoch - self.model_info.current_epoch

        if loss_function is None:
            def loss_func(selected_out, selected_label, theta):
                return nn.BCELoss()(selected_out, selected_label)
            loss_function = loss_func

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in self.get_epoch_iter(
                range(self.model_info.current_epoch, self.model_info.current_epoch + num_epoch)):
            self.print(f"\nEpoch {epoch}")

            """
            Train on train set
            """
            self.model.train()
            self.print(f"Training...")
            sum_train_loss = 0
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.train_loader):
                p_matrix, target_p_matrix, sid = batch
                p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                p_mask = (target_p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                loss = loss_function(selected_out, selected_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            train_loss = sum_train_loss / len(self.train_loader)
            time.sleep(0.1)
            self.print(f"Train Loss : {train_loss}\n")

            """
            Validate on val set
            """
            self.model.eval()
            self.print(f"Validating...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            val_auc = roc_auc_score(cat_label, cat_predict)
            val_acc = accuracy_score(cat_label, np.round(cat_predict))
            val_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Validate AUC : {val_auc}\n")

            """
            Test on test set
            """
            self.print(f"Testing...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            test_auc = roc_auc_score(cat_label, cat_predict)
            test_acc = accuracy_score(cat_label, np.round(cat_predict))
            test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Test AUC : {val_auc}\n")

            self.model_info.add(train_loss, val_auc, test_auc, val_acc, test_acc, val_rmse, test_rmse)
            if self.model_info.is_best():
                self.save_model()
            if epoch % 10 == 0 and epoch != 0 and self.show_plot:
                self.plot()
        # print(f"Val best auc: {self.model_info.best()[0]}\nTest best auc: {self.model_info.best()[1]}")
        if not mute_result:
            print(f"""
            Val best
            Acc: {self.model_info.best()[2]}
            Auc: {self.model_info.best()[0]}
            Rmse: {self.model_info.best()[4]}
            Test best 
            Acc: {self.model_info.best()[3]}
            Auc: {self.model_info.best()[1]}
            Rmse: {self.model_info.best()[5]}""")

    def train_seen(self, num_epoch=None, to_epoch=None, loss_function=None, mask_ratio=0.6, mute_result=False):
        assert self.model is not None
        assert self.train_loader is not None
        if to_epoch is not None:
            num_epoch = to_epoch - self.model_info.current_epoch

        if loss_function is None:
            def loss_func(selected_out, selected_label, theta):
                return nn.BCELoss()(selected_out, selected_label)
            loss_function = loss_func

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in self.get_epoch_iter(
                range(self.model_info.current_epoch, self.model_info.current_epoch + num_epoch)):
            self.print(f"\nEpoch {epoch}")

            """
            Train on train set
            """
            self.model.train()
            self.print(f"Training...")
            sum_train_loss = 0
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.train_loader):
                p_matrix, target_p_matrix, sid = batch
                p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)

                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                p_mask = (target_p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                loss = loss_function(selected_out, selected_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            for loader in [self.val_loader, self.test_loader]:
                for batch in self.get_batch_iter(loader):
                    target_p_matrix, _, sid = batch
                    p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)

                    if self.gpu:
                        p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                    p_mask = (target_p_matrix != 0)

                    out, theta = self.model(p_matrix, sid)
                    selected_out = torch.masked_select(out, mask=p_mask)
                    selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                    loss = loss_function(selected_out, selected_label, theta)
                    if not torch.isnan(loss):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        sum_train_loss += loss.item()

            train_loss = sum_train_loss / (len(self.train_loader) + len(self.val_loader) + len(self.test_loader))
            time.sleep(0.1)
            self.print(f"Train Loss : {train_loss}\n")

            """
            Validate on val set
            """
            self.model.eval()
            self.print(f"Validating...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            val_auc = roc_auc_score(cat_label, cat_predict)
            val_acc = accuracy_score(cat_label, np.round(cat_predict))
            val_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Validate AUC : {val_auc}\n")

            """
            Test on test set
            """
            self.print(f"Testing...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            test_auc = roc_auc_score(cat_label, cat_predict)
            test_acc = accuracy_score(cat_label, np.round(cat_predict))
            test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Test AUC : {val_auc}\n")

            self.model_info.add(train_loss, val_auc, test_auc, val_acc, test_acc, val_rmse, test_rmse)
            if self.model_info.is_best():
                self.save_model()
            if epoch % 10 == 0 and epoch != 0 and self.show_plot:
                self.plot()

        if not mute_result:
            print(f"""
            Val best
            Acc: {self.model_info.best()[2]}
            Auc: {self.model_info.best()[0]}
            Rmse: {self.model_info.best()[4]}
            Test best 
            Acc: {self.model_info.best()[3]}
            Auc: {self.model_info.best()[1]}
            Rmse: {self.model_info.best()[5]}""")


    def train_traditional(self, num_epoch=None, to_epoch=None, loss_function=None):
        assert self.model is not None
        assert self.train_loader is not None
        if to_epoch is not None:
            num_epoch = to_epoch - self.model_info.current_epoch

        if loss_function is None:
            def loss_func(selected_out, selected_label, theta):
                return nn.BCELoss()(selected_out, selected_label)
            loss_function = loss_func

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in self.get_epoch_iter(
                range(self.model_info.current_epoch, self.model_info.current_epoch + num_epoch)):
            self.print(f"\nEpoch {epoch}")
            self.print(f"Training...")
            sum_train_loss = 0
            time.sleep(0.1)

            self.model.train()
            for batch in self.get_batch_iter(self.train_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (target_p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(target_p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)

                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()


            train_loss = sum_train_loss / (len(self.train_loader) + len(self.val_loader) + len(self.test_loader))
            time.sleep(0.1)
            self.print(f"Train Loss : {train_loss}\n")

            """
            Validate on val set
            """
            self.model.eval()
            self.print(f"Validating...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)

            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label)
            cat_predict = np.array(cat_predict)
            try:
                val_auc = roc_auc_score(cat_label, cat_predict)
                val_acc = accuracy_score(cat_label, np.round(cat_predict))
                val_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
                time.sleep(0.1)
                self.print(f"Validate AUC : {val_auc}\n")
            except:
                val_auc, val_acc, val_rmse = 0, 0, 100
                time.sleep(0.1)
                self.print(f"Validate AUC : None\n")

            """
            Test on test set
            """
            self.print(f"Testing...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)

            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                cat_label += selected_label.detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label)
            cat_predict = np.array(cat_predict)
            test_auc = roc_auc_score(cat_label, cat_predict)
            test_acc = accuracy_score(cat_label, np.round(cat_predict))
            test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Test AUC : {test_auc}\n")

            self.model_info.add(train_loss, val_auc, test_auc, val_acc, test_acc, val_rmse, test_rmse)
            if self.model_info.is_best():
                self.save_model()
            if epoch % 10 == 0 and epoch != 0 and self.show_plot:
                self.plot()
        print(f"""
        Val best
        Acc: {self.model_info.best()[2]}
        Auc: {self.model_info.best()[0]}
        Rmse: {self.model_info.best()[4]}
        Test best 
        Acc: {self.model_info.best()[3]}
        Auc: {self.model_info.best()[1]}
        Rmse: {self.model_info.best()[5]}""")

    def train_traditional_unseen(self, num_epoch=None, to_epoch=None, loss_function=None, target_epoch=10, stu_emb_only=False):
        assert self.model is not None
        assert self.train_loader is not None
        if to_epoch is not None:
            num_epoch = to_epoch - self.model_info.current_epoch

        if loss_function is None:
            def loss_func(selected_out, selected_label, theta):
                return nn.BCELoss()(selected_out, selected_label)
            loss_function = loss_func

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        stu_optimizer = optim.Adam(self.model.assess_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in self.get_epoch_iter(
                range(self.model_info.current_epoch, self.model_info.current_epoch + num_epoch)):
            self.print(f"\nEpoch {epoch}")
            self.print(f"Training...")
            sum_train_loss = 0
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.train_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (target_p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(target_p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sum_train_loss += loss.item()

            train_loss = sum_train_loss / (len(self.train_loader) + len(self.val_loader))
            time.sleep(0.1)
            self.print(f"Train Loss : {train_loss}\n")

            """
            Validate on val set
            """
            self.model.eval()
            self.print(f"Validating...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)

            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label)
            cat_predict = np.array(cat_predict)
            try:
                val_auc = roc_auc_score(cat_label, cat_predict)
                val_acc = accuracy_score(cat_label, np.round(cat_predict))
                val_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
                time.sleep(0.1)
                self.print(f"Validate AUC : {val_auc}\n")
            except:
                val_auc, val_acc, val_rmse = 0, 0, 100
                time.sleep(0.1)
                self.print(f"Validate AUC : None\n")

            # self.model_info.add(train_loss, val_auc, 0)
            self.model_info.add(train_loss, val_auc, 0, val_acc, 0, val_rmse, 100)
            if self.model_info.is_best():
                self.save_model()
            if epoch % 10 == 0 and epoch != 0 and self.show_plot:
                self.plot()
        print(f"""
        Val best
        Acc: {self.model_info.best()[2]}
        Auc: {self.model_info.best()[0]}
        Rmse: {self.model_info.best()[4]}""")

        self.load_model()
        optimizer = optimizer if stu_emb_only is False else stu_optimizer
        for epoch in self.get_epoch_iter(range(target_epoch)):
            sum_train_loss = 0
            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()

                loss_mask = (p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                loss_out = torch.masked_select(out, mask=loss_mask)
                loss_label = torch.masked_select(p_matrix, mask=loss_mask) - 1

                loss = loss_function(loss_out, loss_label, theta)

                if not torch.isnan(loss):
                    sum_train_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            self.print(f"Train Loss : {sum_train_loss/len(self.test_loader)}\n")

            """
            Test on test set
            """
            self.print(f"Testing...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)

            stu_aucs = []
            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                stu_auc = roc_auc_score(selected_label.unsqueeze(-1).detach().cpu().numpy(),
                                        selected_out.unsqueeze(-1).detach().cpu().numpy())
                stu_aucs.append(stu_auc)

                cat_label += selected_label.detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()
            print("AUC STD:", np.std(stu_aucs))

            cat_label = np.array(cat_label)
            cat_predict = np.array(cat_predict)
            test_auc = roc_auc_score(cat_label, cat_predict)
            test_acc = accuracy_score(cat_label, np.round(cat_predict))
            test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Test AUC : {test_auc}\n")
            self.model_info.add(0, 0, test_auc, 0, test_acc, 100, test_rmse)
        print(f"""
        Val best
        Acc: {self.model_info.best()[2]}
        Auc: {self.model_info.best()[0]}
        Rmse: {self.model_info.best()[4]}
        Test best 
        Acc: {self.model_info.best_target()[0]}
        Auc: {self.model_info.best_target()[1]}
        Rmse: {self.model_info.best_target()[2]}""")
        return self.model_info.best_target()[0], self.model_info.best_target()[1], self.model_info.best_target()[2]

    def test(self, mask_ratio=0.2):
        """
        Test on test set
        """
        self.print(f"Testing...")
        cat_label, cat_predict = [], []
        time.sleep(0.1)
        response_num_arr = []
        stu_aucs = []
        for batch in self.get_batch_iter(self.test_loader):
            p_matrix, target_p_matrix, sid = batch
            if self.gpu:
                p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
            p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)
            response_num = (torch.count_nonzero(p_matrix).item())/32
            response_num_arr.append(response_num)
            p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

            out, theta = self.model(p_matrix, sid)
            selected_out = torch.masked_select(out, mask=p_mask)
            selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

            cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
            cat_predict += selected_out.detach().cpu().tolist()
            stu_auc = roc_auc_score(selected_label.unsqueeze(-1).detach().cpu().numpy(), selected_out.unsqueeze(-1).detach().cpu().numpy())
            stu_aucs.append(stu_auc)

        stu_auc_std = np.std(stu_aucs)

        avg_response_num = sum(response_num_arr) / len(response_num_arr)
        cat_label = np.array(cat_label)
        cat_predict = np.array(cat_predict)
        test_auc = roc_auc_score(cat_label, cat_predict)
        test_acc = accuracy_score(cat_label, np.round(cat_predict))
        test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
        self.print(f"Test auc: {test_auc}, mask_rate = {mask_ratio}, average response num = {avg_response_num}")
        return test_acc, test_auc, test_rmse, stu_auc_std

    def print(self, x):
        if self.verbose:
            print(x)

    def get_batch_iter(self, x):
        if self.verbose:
            return tqdm(x)
        else:
            return x

    def get_epoch_iter(self, x):
        if self.verbose:
            return x
        else:
            return tqdm(x)

    def plot(self):
        self.model_info.plot(title=self.model_name)

    def print_result(self):
        print(f"""
        Val best
        Acc: {self.model_info.best()[2]}
        Auc: {self.model_info.best()[0]}
        Rmse: {self.model_info.best()[4]}
        Test best 
        Acc: {self.model_info.best()[3]}
        Auc: {self.model_info.best()[1]}
        Rmse: {self.model_info.best()[5]}""")
