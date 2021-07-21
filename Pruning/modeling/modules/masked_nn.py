# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ["topK", "threshold",
                                  "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.init_mask()

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-
                          self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    def forward(self, input: torch.tensor, threshold: float):
        # Get the mask
        # print(f"\n\n\n threshold: {threshold}")
        if self.pruning_method == "topK":
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if self.training:
                u = torch.zeros_like(
                    self.mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid(
                    (u.log() - (1 - u).log() + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Mask weights with computed mask
        weight_thresholded = mask * self.weight
        # Compute output (linear layer) with masked weights
        return F.linear(input, weight_thresholded, self.bias)


class MaskedConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK"

    """

    def __init__(
        self, nf, nx,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        pruning_method: str = "topK",
    ):
        super(MaskedConv1D, self).__init__()
        self.nf = nf
        assert pruning_method in ["topK", "threshold",
                                  "sigmoied_threshold", "magnitude", "l0"]
        self.pruning_method = pruning_method

        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

        if self.pruning_method in ["topK", "threshold", "sigmoied_threshold", "l0"]:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.init_mask()

    def init_mask(self):
        if self.mask_init == "constant":
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(self.mask_scores, a=-
                          self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))


    def forward(self, x, threshold:float):
        # Get the mask
        # print(f"\n\n\n threshold: {threshold}")
        if self.pruning_method == "topK":
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == "magnitude":
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if self.training:
                u = torch.zeros_like(
                    self.mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid(
                    (u.log() - (1 - u).log() + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        # Mask weights with computed mask
        weight_thresholded = mask * self.weight
        # Compute output (linear layer) with masked weights

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight_thresholded)
        x = x.view(*size_out)
        return x
