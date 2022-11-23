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
Count remaining (non-zero) weights in the encoder (i.e. the transformer layers).
Sparsity and remaining weights levels are equivalent: sparsity % = 100 - remaining weights %.
"""
import argparse
import os

import torch

from modeling.modules import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer
from modeling import MaskedGPT2Config, MaskedGPT2LMHeadModel

def count_rem_params(args):
    serialization_file = args.serialization_file
    pruning_method = args.pruning_method
    threshold = args.threshold

    st = torch.load(serialization_file, map_location="cpu")
    config = MaskedGPT2Config(pruning_method=args.pruning_method)
    model = MaskedGPT2LMHeadModel(config=config)
    
    # print("Total num parameters: ", len(list(model.named_parameters())))
    remaining_count = 0  # Number of remaining (not pruned) params in the encoder
    encoder_count = 0  # Number of params in the encoder

    print("name".ljust(60, " "), "Remaining Weights %", "Remaning Weight")

    for name, param in st['model_state'].items():
        if "mask_scores" in name:
            if pruning_method == "topK":
                mask_ones = TopKBinarizer.apply(param, threshold).sum().item()
            elif pruning_method == "magnitude":
                mask_ones = MagnitudeBinarizer.apply(param, threshold).sum().item()
                # print('mask_size', MagnitudeBinarizer.apply(param, threshold))
            elif pruning_method == "sigmoied_threshold":
                mask_ones = ThresholdBinarizer.apply(param, threshold, True).sum().item()
            elif pruning_method == "l0":
                l, r = -0.1, 1.1
                s = torch.sigmoid(param)
                s_bar = s * (r - l) + l
                mask = s_bar.clamp(min=0.0, max=1.0)
                mask_ones = (mask > 0.0).sum().item()
            else:
                raise ValueError("Unknown pruning method")
            remaining_count += mask_ones
            
            print(name.ljust(60, " "), str(round(100 * mask_ones / param.numel(), 3)).ljust(20, " "), str(mask_ones))
        else:
            encoder_count += param.numel()
            if "bias" in name or "LayerNorm" in name:
                remaining_count += param.numel()
    # print("rem count", remaining_count)
    # print("\nRemaining Weights (global) %: ", 100 * remaining_count / encoder_count)
    print("\nRemaining Enc. Weights (global) %: ", 100 * remaining_count / encoder_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "topK", "magnitude", "sigmoied_threshold"],
        type=str,
        required=True,
        help="Pruning Method (l0 = L0 regularization, magnitude = magnitude pruning, "
        "topK = Movement pruning, sigmoied_threshold = Soft movement pruning",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--serialization_file",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )

    args = parser.parse_args()

    count_rem_params(args)
