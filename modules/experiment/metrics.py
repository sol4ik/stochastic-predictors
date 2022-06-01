"""
Every function returns and prints/plots results of a given metric calculations.

Correctness of predictions:
0. Learning curves (CE loss and accuracy)
1. Confusion matrix

Model certainty:
2. Correct predictions certainty
3. Wrong predictions certainty
4. Rejection cost (add option "idk")

Robustness:
5. Robustness against missing features
6. Robustness against adversarial attacks
7. (?) Dependency of gradient distribution on data distribution

Others:
8. Time to converge
9. Generalization error
10. t-SNE features plot
"""
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import yaml
import pickle

from stochastic_predictors.modules.model.cnn import StochasticCNN


EXPERIMENTS_PATH = "../../experiments/"
EXPERIMENT_FOLDER = "exp-"


def confusion_matrix(target, predictions,
                     title="Confusion matrix", xlabel="predictions", ylabel="target", tlabels=None):
    cm = confusion_matrix(target, predictions)

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(cm, annot=True, fmt="g")
    ax.set_title(title)
    ax.figure.tight_layout()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticklabels(tlabels)
    ax.yaxis.set_ticklabels(tlabels)

    plt.show()


def model_certainty(exp_code):
    with open(os.path.join(EXPERIMENTS_PATH,
                           EXPERIMENT_FOLDER + exp_code + "/config-" + exp_code + ".yaml"),
              "r") as config_file:
        params = dict(yaml.load(config_file, Loader=yaml.FullLoader))
    net = StochasticCNN(params["model"], params["n_for_features"], params["batch_norm"])
    net.eval()

    test_data = FashionMNIST("../data", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=64)

    correct = dict()
    wrong = dict()
    for (data, target) in test_loader:
        y = net.forward(data)
        pred = torch.argmax(y, dim=1)


def generalization_error():
    pass

# def