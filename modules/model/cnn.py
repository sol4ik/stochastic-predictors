import torch
import torch.nn as nn
# from stochastic_predictors.modules.model.stochastic import StochasticSignSTE

from tqdm import tqdm

from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform


a, b = 0, 0.5
base_distribution = Uniform(0, 1)
transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
logistic = TransformedDistribution(base_distribution, transforms)


class StochasticSignSTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0

    def forward(self, x):
        z = logistic.rsample(x.size())
        # z = logistic.rsample(x.size()).to(torch.device("cuda"))
        self.p = torch.sigmoid(x - z)
        out = torch.bernoulli(self.p)
        out[out == 0] = -1
        out = out.detach() + (self.p - self.p.detach())
        return out


class StochasticReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        z = logistic.rsample(x.size()).to(torch.device("cuda"))
        self.out = x - z
        self.out[self.out < 0] = 0
        return self.out

    def backward(self, delta_next):
        self.out[self.out > 0] = 1
        return self.out


class StochasticAdditiveConv2d(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        out = super().forward(x)
        z = logistic.rsample(out.size()).cuda()
        # z = torch.normal(mean=torch.zeros(out.size()), std=.025 * torch.ones(out.size())).cuda()
        return out - z


class StochasticMultiplicativeConv2d(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        z = torch.normal(mean=torch.zeros(x.size()), std=.1 * torch.ones(x.size())).cpu()
        return super().forward(x * z)


BLOCKS = {
    "conv2d": nn.Conv2d,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softmax2d": nn.Softmax2d,
    "logsoftmax": nn.LogSoftmax,
    "dropout": nn.Dropout,
    "dropout2d": nn.Dropout2d,
    "batch_norm": nn.BatchNorm2d,
    "flatten": nn.Flatten,
    "batchnorm2d": nn.BatchNorm2d,
    "step_ste": StochasticSignSTE,
    "stochastic_relu": StochasticReLU,
    "stochastic_add_conv2d": StochasticAdditiveConv2d,
    "stochastic_mul_conv2d": StochasticMultiplicativeConv2d
}


class StochasticCNN(nn.Sequential):
    def __init__(self, blocks, n_for_features, if_batch_norm):
        """
        :param blocks: dict - info on net layers from .yaml config file, "model" block
        """
        layers = list()

        in_channels = blocks[0]["channels"]
        out_channels = 0
        for block in tqdm(blocks[1:]):  # first block - input layer parameters
            layer_type = block["type"]

            kwargs = block
            kwargs.pop("type", None)
            if "conv2d" in layer_type:
                out_channels = block["channels"]
                kwargs.pop("channels", None)

                kwargs["in_channels"] = in_channels
                kwargs["out_channels"] = out_channels

            if layer_type not in BLOCKS:
                raise KeyError("invalid block type given!")

            layers.append(BLOCKS[layer_type](**kwargs))
            if if_batch_norm and layer_type == "conv2d":
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        super().__init__(*layers)
        self.layers = layers

        self.n_for_features = n_for_features

    def features(self, x):
        """
        Get features for some data sample.
        :param x: torch.Tensor - data sample/batch
        :return: torch.Tensor
        """
        feats = nn.Sequential(*self.layers[:self.n_for_features + 1]).forward(x)
        feats = torch.flatten(feats, start_dim=1)
        feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats
