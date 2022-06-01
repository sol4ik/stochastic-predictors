import torch
import torch.nn as nn

from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform


a, b = 0, 1
base_distribution = Uniform(0, 1)
transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
logistic = TransformedDistribution(base_distribution, transforms)


class StochasticSignSTE(nn.Module):
    def __init__(self):
        self.p = 0

    def forward(self, x):
        z = logistic.rsample(x.size())
        self.p = torch.sigmoid(x - z)
        out = torch.berboulli(self.p)
        out[out == 0] = -1
        out = out.detach() + (self.p - self.p.detach())
        return out

    def backward(self):
        return self.p


class StochasticReLU(nn.Module):
    def __init__(self):
        self.out = torch.zeros()

    def forward(self, x):
        z = logistic.rsample(x.size())
        self.out = x - z
        self.out[self.out < 0] = 0
        return self.out

    def backward(self):
        self.out[self.out > 0] = 1
        return self.out
