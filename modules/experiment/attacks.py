import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

from torchattacks.attack import Attack


class MissingFeaturesAttack(Attack):
    def __init__(self, model, p):
        super().__init__("MissingFeatures", model)
        self.p = p
        self.dropout = nn.Dropout(p=p)
        self.set_mode_default()

    def forward(self, data, labels=None):
        return self.dropout(data)


def attack(net, dev, ts, attack):
    """
    Targeted Fast Gradient Sign Method attack.
    :param net: torch.nn.Sequential - model to test
    :param dev: torch.device
    :param ts: transforms - data transforms to apply to test data
    :param attack: torchattacks.Attack - attack to perform on the data
    :return: float - model's accuracy on adversarial examples
    :param net:
    :param dev:
    :param ts:
    :param eps:
    :return:
    """
    dataset = FashionMNIST("../data", download=True, train=False, transform=ts)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    net.to(dev)
    net.eval()

    softmax = nn.Softmax(dim=1)
    correct, total = 0, 0
    for data, target in data_loader:
        # if attack.get_mode() == "default":  # untargeted attack
        #     data = attack(data).to(dev)
        # else:
        data = attack(data, target).to(dev)
        out = softmax(net.forward(data))
        _, predicted = torch.max(out.data, 1)

        total += target.size(0)
        correct += (predicted.detach().cpu() == target).sum()
    return correct / total

