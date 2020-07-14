from typing import Dict

import torch
from torch import nn

from configs import EncoderConfig
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES


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
        self.num_directions = 2 if config.use_bi_rnn else 1

        self.subtoken_embedding = nn.Embedding(n_subtokens, config.embedding_size, padding_idx=subtoken_pad_id)
        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)

        # TF apply RNN dropout on inputs, but Torch apply it to the outputs except lasts
        # So, manually adding dropout for the first layer
        self.rnn_dropout = nn.Dropout(config.rnn_dropout)
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

    def forward(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [max name parts; total paths]
        from_token = samples[FROM_TOKEN]
        to_token = samples[TO_TOKEN]
        # [max path length + 1; total paths]
        path_types = samples[PATH_TYPES]

        # [total paths; embedding size]
        encoded_from_tokens = self.subtoken_embedding(from_token).sum(0)
        encoded_to_tokens = self.subtoken_embedding(to_token).sum(0)

        # [max path length + 1; total paths; embedding size]
        path_types_embed = self.type_embedding(path_types)
        # [1 or 2 * num_layers; total paths; rnn size]
        _, (hidden_states, _) = self.path_lstm(self.rnn_dropout(path_types_embed))
        # [total_paths; rnn size * 2]
        encoded_paths = hidden_states[-self.num_directions :].transpose(0, 1).reshape(hidden_states.shape[1], -1)

        # [total_paths; 2 * embedding size + rnn size (*2)]
        concat = torch.cat([encoded_from_tokens, encoded_paths, encoded_to_tokens], dim=-1)
        concat = self.dropout(concat)

        # [total_paths; output size]
        output = self.tanh(self.linear(concat))
        return output
