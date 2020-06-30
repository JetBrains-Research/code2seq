from typing import List

import torch
from torch import nn

from configs import DecoderConfig
from dataset import Vocabulary
from utils.common import segment_sizes_to_slices


class PathDecoder(nn.Module):

    _negative_value = -1e9

    def __init__(self, config: DecoderConfig, vocab: Vocabulary, encoder_size: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.config = config

        self.target_embedding = nn.Embedding(len(vocab.label_to_id), config.decoder_size)
        self.decoder_lstm = nn.LSTM(
            config.decoder_size + encoder_size, encoder_size, num_layers=config.num_decoder_layers
        )
        self.attention = nn.Linear(encoder_size, config.decoder_size)
        self.projection_layer = nn.Linear(config.decoder_size, self.out_size)

    def forward(
        self, encoded_paths: torch.Tensor, target_labels: torch.Tensor, paths_for_label: List[int]
    ) -> torch.Tensor:
        """Decode given paths into sequence

        :param encoded_paths: [n paths; encoder size]
        :param target_labels: [max output length; batch size]
        :param paths_for_label: [n1, n2, ..., nk] sum = n paths
        :return:
        """
        max_output_length, batch_size = target_labels.shape

        # [batch size; context size; encoder size]
        max_context_per_batch = max(paths_for_label)
        batched_context = encoded_paths.new_zeros(batch_size, max_context_per_batch, encoded_paths.shape[1])
        attention_mask = encoded_paths.new_zeros((batch_size, max_context_per_batch), dtype=torch.bool)
        for i, (cur_slice, cur_size) in enumerate(zip(segment_sizes_to_slices(paths_for_label), paths_for_label)):
            batched_context[i, :cur_size] = encoded_paths[cur_slice]
            attention_mask[i, cur_size:] = True
        # [batch size; context size; decoder size]
        attended_batched_context = self.attention(batched_context)

        # [batch size; encoder size]
        initial_state = batched_context.sum(dim=1) / paths_for_label
        # [n layers; batch size; encoder size]
        h_prev = initial_state.unsqueeze(0).repeat(self.config.num_decoder_layers, 1, 1)
        c_prev = initial_state.unsqueeze(0).repeat(self.config.num_decoder_layers, 1, 1)

        # [target len; batch size; vocab size]
        output = encoded_paths.new_zeros((max_output_length, batch_size, self.out_size))
        # [batch size] (first row always consists of <SOS> tokens)
        current_input = target_labels[0:]
        for step in range(1, max_output_length):
            # 1. calculate attention weights.
            # [batch size; context size]
            attn_weights = torch.bmm(attended_batched_context, h_prev[-1].unsqueeze(1))
            attn_weights[attention_mask] = self._negative_value
            scores = nn.functional.softmax(attn_weights, dim=-1)

            # 2. apply scores to batched context.
            # [batch size; encoder size]
            attended_context = (batched_context * scores).sum(1)

            # 3. prepare lstm input.
            input_embedding = self.target_embedding(current_input)
            # [1; batch size; decoder size + encoder size]
            lstm_input = torch.cat([input_embedding, attended_context]).unsqueeze(1)

            # 4. do decoder step.
            # [1; batch size; encoder size]
            lstm_output, (h_prev, c_prev) = self.decoder_lstm(lstm_input, (h_prev, c_prev))

            # 5. project result and prepare to the next step.
            output[step] = self.projection_layer(lstm_output.squeeze(1))
            current_input = output[step].argmax(dim=-1)

        return output
