import os
import pickle
import yaml

import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from modules.model.cnn import StochasticCNN


EXPERIMENTS_PATH = "../../experiments/"
EXPERIMENT_FOLDER = "exp-"
CLASSES = [
    "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def print_experiment(exp_code):
    """
    Print out experiment details.
    :param exp_code: str - experiment code, e.g. b01 = baseline model, experiment #1
    """
    with open(os.path.join(EXPERIMENTS_PATH,
                           EXPERIMENT_FOLDER + exp_code + "/config-" + exp_code + ".yaml"),
              "r") as config_file:
        params = dict(yaml.load(config_file, Loader=yaml.FullLoader))
    print("Model architecture:")
    net = StochasticCNN(params["model"], params["n_for_features"], params["batch_norm"])
    print(net)
    print("=========================")
    print("Training loop parameters:")
    print("number of epochs: {}\nlearning rate: {}\noptimizer: Adam\nbatch size: {}".format(
        params["n_epochs"],
        params["learning_rate"],
        params["natch_size"]
    ))
    params["normalize_params"] = list(map(float, params["normalize_params"].strip().split(",")))
    print("=========================")
    print("Data normalization parameters:")
    print("mean: {:.3f}\nstandart deviation: {:.3f}".format(params["normalize_params"][0], params["normalize_params"][1]))
    print("=========================")


def plot_history(exp_code):
    """
    Plot training curves.
    :param exp_code: str - experiment code, e.g. b01 = baseline model, experiment #1
    """
    with open(os.path.join(EXPERIMENTS_PATH,
                           EXPERIMENT_FOLDER + exp_code + "/config-" + exp_code + ".yaml"),
              "r") as config_file:
        params = dict(yaml.load(config_file, Loader=yaml.FullLoader))
    with open(os.path.join(
            params["res_path"], params["basename"] + params["call_suffix"] + "-history.pickle"
    )) as history:
        history = pickle.load(history)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if params["objective"] == "likelihood":
        axs[0].plot(-1 * history["train_obj"], label="train")
        axs[0].plot(-1 * history["val_obj"], label="validation")
        axs[0].set_title("Likelihood")
    else:
        axs[0].plot(history["train_obj"], label="train")
        axs[0].plot(history["val_obj"], label="validation")
        axs[0].set_title("Cross Entropy")
    axs[0].set_xlabel("epoch")
    axs[0].legend()
    axs[1].plot(history["train_acc"], label="train")
    axs[1].plot(history["val_acc"], label="validation")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].legend()

    plt.show()


def plot_predictions(exp_code):
    """
    Illustrate sample data with respective model predictions.
    :param exp_code: str - experiment code, e.g. b01 = baseline model, experiment #1
    """
    with open(os.path.join(EXPERIMENTS_PATH,
                           EXPERIMENT_FOLDER + exp_code + "/config-" + exp_code + ".yaml"),
              "r") as config_file:
        params = dict(yaml.load(config_file, Loader=yaml.FullLoader))
    net = StochasticCNN(params["model"], params["n_for_features"], params["batch_norm"])
    net.eval()
    softmax = nn.Sotmax(dim=1)

    test_data = FashionMNIST("../data", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=1)

    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    for i, (data, target) in enumerate(test_loader):
        y = softmax(net.forward(data))
        pred = torch.argmax(y, dim=1).item()
        axs.flat[i].inshow(data.permute(1, 2, 0), cmap="gray")
        axs.flat[i].axis("off")
        axs.flat[i].set_title(f"{CLASSES[pred]}: {y[pred]:.2f}")
    plt.show()


def plot_features_tsne(exp_code):
    """
    Scatter tSNE plot of extracted features.
    :param exp_code: str - experiment code, e.g. b01 = baseline model, experiment #1
    """
    with open(os.path.join(EXPERIMENTS_PATH,
                           EXPERIMENT_FOLDER + exp_code + "/config-" + exp_code + ".yaml"),
              "r") as config_file:
        params = dict(yaml.load(config_file, Loader=yaml.FullLoader))
    net = StochasticCNN(params["model"], params["n_for_features"], params["batch_norm"])
    net.eval()

    test_data = FashionMNIST("../data", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=5000)

    for (data, target) in test_loader:
        features = net.features(data).flatten(start_dim=1)
    features = features.cpu().detach().numpy()

    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    plt.clf()
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'yellow', 'orange', 'purple']
    for i in range(10):
        idx = np.where(target == i)[0]
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors[i], label=CLASSES[i])
    plt.xlabel("x [units]")
    plt.ylabel("y [units]")
    plt.title("tSNE plot of features extracted by model")
    plt.legend()
    plt.show()
