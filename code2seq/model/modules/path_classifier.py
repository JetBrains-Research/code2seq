from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from code2seq.utils.training import cut_encoded_contexts
from .attention import LocalAttention


class PathClassifier(nn.Module):

    _negative_value = -1e9
    _activations = {"relu": torch.nn.ReLU(), "sigmoid": torch.nn.Sigmoid(), "tanh": torch.nn.Tanh()}

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name in self._activations:
            return self._activations[activation_name]
        raise KeyError(f"Activation {activation_name} is not supported")

    def __init__(self, config: DictConfig, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.attention = LocalAttention(config.classifier_input_size)
        layers = [nn.Linear(config.classifier_input_size, config.hidden_size), self._get_activation(config.activation)]
        if config.n_hidden_layers < 1:
            raise ValueError(f"Invalid layers number ({config.n_hidden_layers})")
        for _ in range(config.n_hidden_layers - 1):
            layers += [nn.Linear(config.hidden_size, config.hidden_size), self._get_activation(config.activation)]
        self.hidden_layers = nn.Sequential(*layers)
        self.classification_layer = nn.Linear(config.hidden_size, self.out_size)

    def forward(
        self,
        encoded_paths: torch.Tensor,
        contexts_per_label: List[int],
    ) -> torch.Tensor:
        """Classify given paths

        :param encoded_paths: [n paths; classifier size]
        :param contexts_per_label: [n1, n2, ..., nk] sum = n paths
        :return:
        """
        # [batch size; max context size; classifier input size], [batch size; max context size]
        batched_context, attention_mask = cut_encoded_contexts(encoded_paths, contexts_per_label, self._negative_value)

        # [batch size; max context size; 1]
        attn_weights = self.attention(batched_context, attention_mask)

        # [batch size; classifier input size]
        context = torch.bmm(attn_weights.transpose(1, 2), batched_context).squeeze(1)

        # [batch size; hidden size]
        hidden = self.hidden_layers(context)

        # [batch size; num classes]
        output = self.classification_layer(hidden)
        return output
