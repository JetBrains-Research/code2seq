from typing import List

import torch
from omegaconf import DictConfig
from torch import nn


class PathEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
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

        self.dropout_rnn = nn.Dropout(config.encoder_dropout)
        self.path_lstm = nn.LSTM(
            config.embedding_size,
            config.encoder_rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.encoder_dropout if config.rnn_num_layers > 1 else 0,
        )

        concat_size = self._calculate_concat_size(config.embedding_size, config.encoder_rnn_size, self.num_directions)
        self.embedding_dropout = nn.Dropout(config.encoder_dropout)
        if "decoder_size" in config:
            out_size = config["decoder_size"]
        elif "classifier_size" in config:
            out_size = config["classifier_size"]
        else:
            raise ValueError("Specify out size of encoder")
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.norm = nn.LayerNorm(out_size)

    @staticmethod
    def _calculate_concat_size(embedding_size: int, rnn_size: int, num_directions: int) -> int:
        return embedding_size * 2 + rnn_size * num_directions

    def _token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens).sum(0)

    def _path_nodes_embedding(self, path_nodes: torch.Tensor) -> torch.Tensor:
        # [max path length; n contexts; embedding size]
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
        # [n contexts; sum across all embeddings]
        concat = torch.cat(encoded_contexts, dim=-1)

        # [n contexts; output size]
        concat = self.embedding_dropout(concat)
        return torch.tanh(self.norm(self.linear(concat)))

    def forward(self, from_token: torch.Tensor, path_nodes: torch.Tensor, to_token: torch.Tensor) -> torch.Tensor:
        """Encode each path context into the vector

        :param from_token: [max token parts; n contexts] start tokens
        :param path_nodes: [path length; n contexts] path nodes
        :param to_token: [max tokens parts; n contexts] end tokens
        :return: [n contexts; encoder size]
        """
        # [n contexts; embedding size]
        encoded_from_tokens = self._token_embedding(from_token)
        encoded_to_tokens = self._token_embedding(to_token)

        # [n contexts; rnn size * num directions]
        encoded_paths = self._path_nodes_embedding(path_nodes)

        # [n contexts; output size]
        output = self._concat_with_linear([encoded_from_tokens, encoded_paths, encoded_to_tokens])
        return output
