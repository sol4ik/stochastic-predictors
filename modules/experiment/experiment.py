import torch
from torch import nn, optim
import torchvision.transforms as transforms

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

import numpy as np

import os
import yaml
import pickle
from functools import partial

from model.cnn import StochasticCNN


class StochasticExperiment:
    """
    Interface for training a segmentation CNN.
    """
    OBJECTIVES = {
        "nll": nn.NLLLoss,
        "ce": nn.CrossEntropyLoss
    }

    METRICS = {
        "likelihood": None,
        "accuracy": None
    }

    def __init__(self, config):
        """
        :param config: str - path to .yaml config file
        """
        self.config = config
        self.params = None
        self.device = None
        self.transforms = None
        self.train_dataset, self.train_loader = None, None
        self.val_dataset, self.val_loader = None, None
        self.net = None
        self.optimizer, self.loss = None, None
        self.history, self.model = None, None
        self.activations = dict()

    def save_activation_stats(self, name, module, input, out):
        if name not in self.activations:
            self.activations[name] = list()
        mean, std = out.detach().cpu().mean(), out.detach().cpu().std()
        self.activations[name].append((mean, std))

    def set_experiment(self):
        print("reading config file...")
        with open(self.config, "r") as config_file:
            self.params = dict(yaml.load(config_file, Loader=yaml.FullLoader))

        print("setting up the experiment...")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.params["normalize_params"] = list(map(float, self.params["normalize_params"].strip().split(",")))
        self.params["normalize_params"] = {
            "mean": self.params["normalize_params"][0],
            "std": self.params["normalize_params"][1]
        }

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.params["normalize_params"])
        ])

        self.train_dataset = FashionMNIST("../data", download=True, train=True, transform=self.transforms)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        self.val_dataset = FashionMNIST("../data", download=False, train=False, transform=self.transforms)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.params["batch_size"], shuffle=True)

        print("building model...")
        self.net = StochasticCNN(self.params["model"], self.params["n_for_features"], self.params["batch_norm"])
        self.net = self.net.to(self.device)
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(partial(self.save_activation_stats, name))

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.params["learning_rate"])
        self.loss = StochasticExperiment.OBJECTIVES[self.params["objective"]](reduction="none")

        self.history = {
            key: list() for key in ["train_obj", "val_obj", "train_acc", "val_acc"]
        }

    def run_training(self):
        """
        Training loop with validation phase.
        """
        print("\nmodel:")
        print(self.net)

        print("\nusing device ", self.device)

        print("\ntraining...\n")
        softmax = nn.Softmax(dim=1)
        for epoch in range(self.params["n_epochs"]):
            print(f'Epoch: {epoch + 1} / {self.params["n_epochs"]}')
            train_obj = 0
            train_acc = 0
            n_data = 0

            self.net.train()
            for i, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                y = softmax(self.net.forward(data))
                l = self.loss(y, target)
                train_obj += l.sum().item()

                pred = torch.argmax(y, dim=1)
                train_acc += torch.sum(target == pred)
                n_data += data.size(0)

                self.optimizer.zero_grad()
                l.mean().backward()
                self.optimizer.step()
            train_obj /= n_data
            train_acc = float(train_acc) / n_data

            if self.val_loader is not None:
                val_obj = 0
                val_acc = 0
                n_data = 0
                self.net.eval()
                for i, (data, target) in enumerate(self.val_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    y = softmax(self.net.forward(data))
                    l = self.loss(y, target)
                    val_obj += l.sum().item()
                    pred = torch.argmax(y, dim=1)
                    val_acc += torch.sum(target == pred)

                    n_data += data.size(0)
                val_obj /= n_data
                val_acc = float(val_acc) / n_data

            print(f"\ttrain loss: {train_obj}, val loss: {val_obj}")
            print(f"\ttrain accuracy {train_acc}, val accuracy: {val_acc}")
            self.history["train_obj"].append(train_obj)
            self.history["val_obj"].append(val_obj)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if epoch % 25 == 0 and epoch > 0:
                print("saving model checkpoint...")
                j = epoch // 25
                torch.save(self.net.state_dict(),
                           os.path.join(self.params["res_path"],
                                        self.params["basename"] + self.params["call_suffix"] + "-checkpoint" + str(j) + ".pt"))

        print("\nsaving history...")
        history_file_p = os.path.join(self.params["res_path"],
                                    self.params["basename"] + self.params["call_suffix"] + "-history.pickle")
        with open(history_file_p, 'wb') as history_file:
            pickle.dump(self.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

        act_file_p = os.path.join(self.params["res_path"], self.params["basename"] + self.params["call_suffix"] + "-act.pickle")
        with open(act_file_p, 'wb') as activations_file:
            pickle.dump(self.activations, activations_file, protocol=pickle.HIGHEST_PROTOCOL)

        print("saving model...")
        torch.save(self.net.state_dict(),
                   os.path.join(self.params["res_path"], self.params["basename"] + self.params["call_suffix"] + "-statedict.pt"))

        print("\ndone!")
