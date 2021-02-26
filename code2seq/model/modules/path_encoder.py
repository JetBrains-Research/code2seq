from typing import Dict, List

import torch
from omegaconf import DictConfig
from torch import nn

from code2seq.dataset.data_classes import FROM_TOKEN, TO_TOKEN, PATH_NODES


class PathEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        out_size: int,
        n_tokens: int,
        token_pad_id: int,
        n_nodes: int,
        node_pad_id: int,
    ):
        super().__init__()
        self.node_pad_id = node_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1

        self.token_embedding = nn.Embedding(n_tokens, config.embedding_size, padding_idx=token_pad_id)
        self.node_embedding = nn.Embedding(n_nodes, config.embedding_size, padding_idx=node_pad_id)

        self.dropout_rnn = nn.Dropout(config.rnn_dropout)
        self.path_lstm = nn.LSTM(
            config.embedding_size,
            config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )

        concat_size = self._calculate_concat_size(config.embedding_size, config.rnn_size, self.num_directions)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.norm = nn.LayerNorm(out_size)

    @staticmethod
    def _calculate_concat_size(embedding_size: int, rnn_size: int, num_directions: int) -> int:
        return embedding_size * 2 + rnn_size * num_directions

    def _token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens).sum(0)

    def _path_nodes_embedding(self, path_nodes: torch.Tensor) -> torch.Tensor:
        # [max path length; total paths; embedding size]
        path_nodes_embeddings = self.node_embedding(path_nodes)

        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(path_nodes == self.node_pad_id, dim=0)
            first_pad_pos[~is_contain_pad_id] = path_nodes.shape[0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        path_nodes_embeddings = path_nodes_embeddings[:, sort_indices]

        packed_path_nodes = nn.utils.rnn.pack_padded_sequence(path_nodes_embeddings, sorted_path_lengths)

        # [num layers * num directions; total paths; rnn size]
        _, (h_t, _) = self.path_lstm(packed_path_nodes)
        # [total_paths; rnn size * num directions]
        encoded_paths = h_t[-self.num_directions :].transpose(0, 1).reshape(h_t.shape[1], -1)
        encoded_paths = self.dropout_rnn(encoded_paths)

        encoded_paths = encoded_paths[reverse_sort_indices]
        return encoded_paths

    def _concat_with_linear(self, encoded_contexts: List[torch.Tensor]) -> torch.Tensor:
        # [total_paths; sum across all embeddings]
        concat = torch.cat(encoded_contexts, dim=-1)

        # [total_paths; output size]
        concat = self.embedding_dropout(concat)
        return torch.tanh(self.norm(self.linear(concat)))

    def forward(self, contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [total paths; embedding size]
        encoded_from_tokens = self._token_embedding(contexts[FROM_TOKEN])
        encoded_to_tokens = self._token_embedding(contexts[TO_TOKEN])

        # [total_paths; rnn size * num directions]
        encoded_paths = self._path_nodes_embedding(contexts[PATH_NODES])

        # [total_paths; output size]
        output = self._concat_with_linear([encoded_from_tokens, encoded_paths, encoded_to_tokens])
        return output
