from typing import Dict, List

import torch
from torch import nn

from configs import EncoderConfig
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES
from utils.training import create_embedding_tf_style


class PathEncoder(nn.Module):
    def __init__(
        self,
        config: EncoderConfig,
        out_size: int,
        n_subtokens: int,
        subtoken_pad_id: int,
        n_types: int,
        type_pad_id: int,
    ):
        super().__init__()
        self.type_pad_id = type_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1

        self.subtoken_embedding = create_embedding_tf_style(n_subtokens, config.embedding_size, subtoken_pad_id)
        self.type_embedding = create_embedding_tf_style(n_types, config.embedding_size, type_pad_id)

        self.path_lstm = nn.LSTM(
            config.embedding_size,
            config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout,
        )

        self.dropout = nn.Dropout(config.embedding_dropout)

        concat_size = config.embedding_size * 2 + config.rnn_size * self.num_directions
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [max name parts; total paths]
        from_token = contexts[FROM_TOKEN]
        to_token = contexts[TO_TOKEN]

        # [total paths; embedding size]
        encoded_from_tokens = self.subtoken_embedding(from_token).sum(0)
        encoded_to_tokens = self.subtoken_embedding(to_token).sum(0)

        # [max path length; total paths]
        path_types = contexts[PATH_TYPES]
        # [max path length; total paths; embedding size]
        path_types_embed = self.type_embedding(path_types)
        with torch.no_grad():
            path_lengths = (path_types != self.type_pad_id).sum(0)

        # create packed sequence (don't forget to set enforce sorted True for ONNX support)
        packed_path_types = nn.utils.rnn.pack_padded_sequence(path_types_embed, path_lengths, enforce_sorted=False)

        # [num directions; total paths; rnn size]
        _, (h_t, _) = self.path_lstm(packed_path_types)
        # [total_paths; rnn size (*2)]
        encoded_paths = h_t[-self.num_directions :].transpose(0, 1).reshape(h_t.shape[1], -1)

        # [total_paths; 2 * embedding size + rnn size (*2)]
        concat = torch.cat([encoded_from_tokens, encoded_paths, encoded_to_tokens], dim=-1)
        concat = self.dropout(concat)

        # [total_paths; output size]
        output = self.tanh(self.linear(concat))
        return output
