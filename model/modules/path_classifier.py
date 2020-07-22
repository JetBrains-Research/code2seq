from typing import List

import torch
from torch import nn

from configs import ClassifierConfig
from utils.training import cut_encoded_contexts
from .attention import LuongAttention


class PathClassifier(nn.Module):

    _negative_value = -1e9

    def __init__(self, config: ClassifierConfig):
        super().__init__()

        self.attention = LuongAttention(config.classifier_size)

        self.linear = nn.Linear(config.classifier_size, config.num_classes)

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

        # [batch size; num classes]
        output = self.linear(context)

        return output
