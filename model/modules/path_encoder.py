from typing import Dict, Tuple

import torch
from torch import nn

from configs import EncoderConfig
from dataset import Vocabulary
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES, PAD, PATHS_FOR_LABEL


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
        self.subtoken_embedding = nn.Embedding(n_subtokens, config.embedding_size, padding_idx=subtoken_pad_id)

        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)
        self.path_lstm = nn.LSTM(config.embedding_size, config.rnn_size, bidirectional=config.use_bi_rnn)

        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.rnn_dropout = nn.Dropout(config.rnn_dropout)

        concat_size = config.embedding_size * 2 + config.rnn_size * (2 if config.use_bi_rnn else 1)
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [max name parts + 1; total paths]
        from_token = samples[FROM_TOKEN]
        to_token = samples[TO_TOKEN]
        # [max path length + 1; total paths]
        path_types = samples[PATH_TYPES]

        # [total paths; embedding size]
        from_tokens_sum = self.embedding_dropout(self.subtoken_embedding(from_token)).sum(0)
        to_tokens_sum = self.embedding_dropout(self.subtoken_embedding(to_token)).sum(0)

        # [max path length + 1; total paths; embedding size]
        path_types_embed = self.type_embedding(path_types)
        # [max path length + 1; total paths; rnn size (*2)]
        path_types_lstm, (_, _) = self.path_lstm(path_types_embed)
        # [total_paths; rnn size (*2)]
        last_path_state = path_types_lstm[-1]
        last_path_state = self.rnn_dropout(last_path_state)

        # [total_paths; 2 * embedding size + rnn size (*2)]
        concat = torch.cat([from_tokens_sum, last_path_state, to_tokens_sum], dim=-1)

        # [total_paths; output size]
        output = self.tanh(self.linear(concat))
        return output
