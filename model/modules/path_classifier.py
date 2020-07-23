from typing import List

import torch
from torch import nn

from configs import ClassifierConfig
from utils.training import cut_encoded_contexts
from .attention import LuongAttention


class PathClassifier(nn.Module):

    _negative_value = -1e9

    def __init__(self, config: ClassifierConfig, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.attention = LuongAttention(config.classifier_size)
        self.concat_layer = nn.Linear(2 * config.classifier_size, config.hidden_size)
        self.classification_layer = nn.Linear(config.hidden_size, self.out_size)

    def forward(self, encoded_paths: torch.Tensor, contexts_per_label: List[int],) -> torch.Tensor:
        """Classify given paths

        :param encoded_paths: [n paths; classifier size]
        :param contexts_per_label: [n1, n2, ..., nk] sum = n paths
        :return:
        """
        # [batch size; max context size; classifier size], [batch size; max context size]
        batched_context, attention_mask = cut_encoded_contexts(encoded_paths, contexts_per_label, self._negative_value)

        # [batch size; classifier size]
        initial_state = torch.cat(
            [ctx_batch.mean(0).unsqueeze(0) for ctx_batch in encoded_paths.split(contexts_per_label)]
        )
        attn_weights = self.attention(initial_state, batched_context, attention_mask)

        # [batch size; 1; classifier size]
        context = torch.bmm(attn_weights.transpose(1, 2), batched_context)

        # [batch size; classifier size]
        context = context.view(context.shape[0], -1)

        # [batch size; 2 * decoder size]
        concat_input = torch.cat([initial_state, context], dim=2)

        # [batch size; classifier size]
        concat = torch.tanh(self.concat_layer(concat_input))

        # [batch size; num classes]
        output = self.classification_layer(concat)

        return output
