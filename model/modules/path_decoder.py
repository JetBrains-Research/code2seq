from typing import List

import torch
from torch import nn

from configs import DecoderConfig
from utils.common import segment_sizes_to_slices


class PathDecoder(nn.Module):

    _negative_value = -1e9

    def __init__(
        self, config: DecoderConfig, out_size: int, sos_token: int, pad_token: int,
    ):
        super().__init__()
        self.sos_token = sos_token
        self.out_size = out_size
        self.num_decoder_layers = config.num_decoder_layers
        self.teacher_forcing = config.teacher_forcing

        self.target_embedding = nn.Embedding(self.out_size, config.embedding_size, padding_idx=pad_token)
        self.attention = nn.Linear(config.decoder_size, config.decoder_size)

        # TF apply RNN dropout on inputs, but Torch apply it to the outputs except lasts
        # So, manually adding dropout for the first layer
        self.lstm_dropout = nn.Dropout(config.rnn_dropout)
        self.decoder_lstm = nn.LSTM(
            config.decoder_size + config.embedding_size,
            config.decoder_size,
            num_layers=config.num_decoder_layers,
            dropout=config.rnn_dropout,
        )

        self.projection_layer = nn.Linear(config.decoder_size, self.out_size, bias=False)

    def forward(
        self,
        encoded_paths: torch.Tensor,
        contexts_per_label: List[int],
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode given paths into sequence

        :param encoded_paths: [n paths; decoder size]
        :param contexts_per_label: [n1, n2, ..., nk] sum = n paths
        :param output_length: length of output sequence
        :param target_sequence: [sequence length; batch size]
        :return:
        """
        batch_size = len(contexts_per_label)

        max_context_per_batch = max(contexts_per_label)
        # [batch size; context size; decoder size]
        batched_context = encoded_paths.new_zeros(batch_size, max_context_per_batch, encoded_paths.shape[1])
        # [batch size; context size]
        attention_mask = encoded_paths.new_zeros((batch_size, max_context_per_batch), dtype=torch.bool)
        for i, (cur_slice, cur_size) in enumerate(zip(segment_sizes_to_slices(contexts_per_label), contexts_per_label)):
            batched_context[i, :cur_size] = encoded_paths[cur_slice]
            attention_mask[i, cur_size:] = True

        # [batch size]
        contexts_per_label_tensor = encoded_paths.new_tensor(contexts_per_label).view(-1, 1)
        # [batch size; decoder size]
        initial_state = batched_context.sum(dim=1) / contexts_per_label_tensor
        # [n layers; batch size; decoder size]
        h_prev = initial_state.unsqueeze(0).repeat(self.num_decoder_layers, 1, 1)
        c_prev = initial_state.unsqueeze(0).repeat(self.num_decoder_layers, 1, 1)

        # [target len; batch size; vocab size]
        output = encoded_paths.new_zeros((output_length, batch_size, self.out_size))
        # [batch size]
        current_input = torch.full((batch_size,), self.sos_token, dtype=torch.long, device=encoded_paths.device)
        for step in range(1, output_length):
            # 1. calculate attention weights.
            # [batch size; context size; decoder size]
            attended_batched_context = self.attention(batched_context)
            # [batch size; context size]
            attn_weights = torch.bmm(attended_batched_context, h_prev[-1].unsqueeze(1).transpose(1, 2)).squeeze(2)
            attn_weights[attention_mask] = self._negative_value
            # [batch size; context size; 1]
            scores = nn.functional.softmax(attn_weights, dim=-1).unsqueeze(2)

            # 2. apply scores to batched context.
            # [batch size; decoder size]
            attended_context = (batched_context * scores).sum(1)

            # 3. prepare lstm input.
            # [batch size; embedding size]
            input_embedding = self.target_embedding(current_input)
            # [1; batch size; embedding size + decoder size]
            lstm_input = torch.cat([input_embedding, attended_context], dim=1).unsqueeze(0)

            # 4. do decoder step.
            # [1; batch size; decoder size]
            lstm_output, (h_prev, c_prev) = self.decoder_lstm(self.lstm_dropout(lstm_input), (h_prev, c_prev))

            # 5. project result and prepare forK the next step.
            output[step] = self.projection_layer(lstm_output.squeeze(1))

            # 6. prepare next input.
            # if random value is less than teacher_forcing parameter than use target as a label
            # teacher_forcing == 1 is equal to permanent usage of ground truth labels
            if target_sequence is not None and torch.rand(1) < self.teacher_forcing:
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)

        return output
