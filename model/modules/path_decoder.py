from typing import List, Tuple

import torch
from torch import nn

from configs import DecoderConfig
from utils.training import cut_encoded_contexts
from .attention import LuongAttention


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

        self.attention = LuongAttention(config.decoder_size)

        # TF apply RNN dropout on inputs, but Torch apply it to the outputs except lasts
        # So, manually adding dropout for the first layer
        self.lstm_dropout = nn.Dropout(config.rnn_dropout)
        self.decoder_lstm = nn.LSTM(
            config.embedding_size + config.decoder_size,
            config.decoder_size,
            num_layers=config.num_decoder_layers,
            dropout=config.rnn_dropout,
        )

        self.concat_layer = nn.Linear(config.decoder_size * 2, config.decoder_size)
        self.projection_layer = nn.Linear(config.decoder_size, self.out_size)

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
        batched_context, attention_mask = cut_encoded_contexts(encoded_paths, contexts_per_label, self._negative_value)

        # [n layers; batch size; decoder size]
        initial_state = (
            torch.cat([ctx_batch.mean(0).unsqueeze(0) for ctx_batch in encoded_paths.split(contexts_per_label)])
            .unsqueeze(0)
            .repeat(self.num_decoder_layers, 1, 1)
        )
        h_prev, c_prev = initial_state, initial_state

        # [target len; batch size; vocab size]
        output = encoded_paths.new_zeros((output_length, batch_size, self.out_size))
        # [batch size]
        current_input = encoded_paths.new_full((batch_size,), self.sos_token, dtype=torch.long)
        for step in range(1, output_length):
            current_output, (h_prev, c_prev) = self.decoder_step(
                current_input, h_prev, c_prev, batched_context, attention_mask
            )
            output[step] = current_output
            if target_sequence is not None and torch.rand(1) < self.teacher_forcing:
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)

        return output

    def decoder_step(
        self,
        input_tokens: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        batched_context: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # [batch size; embedding size]
        embedded = self.target_embedding(input_tokens)
        # [1; batch size; embedding size] LSTM input required
        embedded = embedded.unsqueeze(0)

        # [batch size; 1; context size]
        attn_weights = self.attention(h_prev[-1], batched_context, attention_mask)

        # [batch size; 1; decoder size]
        context = torch.bmm(attn_weights, batched_context)
        # [batch size; decoder size]
        context = context.squeeze(1)

        # [batch size; embedding size + decoder size]
        lstm_input = torch.cat([embedded, context.unsqueeze(0)], dim=2)

        # [1; batch size; decoder size]
        rnn_output, (h_prev, c_prev) = self.decoder_lstm(lstm_input, (h_prev, c_prev))
        # [batch size; decoder size]
        rnn_output = rnn_output.squeeze(0)

        # [batch size; decoder size]
        # concat = torch.tanh(self.concat_layer(concat_input))

        # [batch size; vocab size]
        output = self.projection_layer(rnn_output)

        return output, (h_prev, c_prev)
