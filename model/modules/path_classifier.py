from typing import List

import torch
from torch import nn

from configs import ClassifierConfig
from utils.training import cut_encoded_contexts
from .attention import LocalAttention


class PathClassifier(nn.Module):

    _negative_value = -1e9
    _activations = {"relu": torch.nn.ReLU(), "sigmoid": torch.nn.Sigmoid(), "tanh": torch.nn.Tanh()}

    def __init__(self, config: ClassifierConfig, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.attention = LocalAttention(config.classifier_input_size)
        layers = [
            nn.Sequential(
                self.activations[config.activation], nn.Linear(config.classifier_input_size, config.hidden_size)
            )
        ]
        layers += [
            nn.Sequential(self.activations[config.activation], nn.Linear(config.hidden_size, config.hidden_size))
            for _ in range(config.n_hidden_layers)
        ]
        self.hidden_layers = nn.Sequential(*layers)
        self.classification_layer = nn.Linear(config.hidden_size, self.out_size)

    def forward(self, encoded_paths: torch.Tensor, contexts_per_label: List[int],) -> torch.Tensor:
        """Classify given paths

        :param encoded_paths: [n paths; classifier size]
        :param contexts_per_label: [n1, n2, ..., nk] sum = n paths
        :return:
        """
        # [batch size; max context size; classifier input size], [batch size; max context size]
        batched_context, attention_mask = cut_encoded_contexts(encoded_paths, contexts_per_label, self._negative_value)

        # [batch size; classifier input size]
        attn_weights = self.attention(batched_context, attention_mask)

        # [batch size; classifier input size]
        context = torch.sum(batched_context * attn_weights, dim=1)

        # [batch size; classifier input size]
        concat = torch.tanh(self.concat_layer(context))

        # [batch size; hidden size]
        hidden = self.hidden_layers(concat)

        # [batch size; num classes]
        output = self.classification_layer(hidden)
        return output