from typing import Dict

import torch
from omegaconf import DictConfig
from torch import nn

from code2seq.dataset.data_classes import FROM_TOKEN, TO_TOKEN, FROM_TYPE, TO_TYPE, PATH_NODES
from code2seq.model.modules import PathEncoder


class TypedPathEncoder(PathEncoder):
    def __init__(
        self,
        config: DictConfig,
        out_size: int,
        n_tokens: int,
        token_pad_id: int,
        n_nodes: int,
        node_pad_id: int,
        n_types: int,
        type_pad_id: int,
    ):
        super().__init__(config, out_size, n_tokens, token_pad_id, n_nodes, node_pad_id)

        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)

    @staticmethod
    def _calculate_concat_size(embedding_size: int, rnn_size: int, num_directions: int) -> int:
        return embedding_size * 4 + rnn_size * num_directions

    def _type_embedding(self, types: torch.Tensor) -> torch.Tensor:
        return self.type_embedding(types).sum(0)

    def forward(self, contexts: Dict[str, torch.Tensor]) -> torch.Tensor:
        # [total paths; embedding size]
        encoded_from_tokens = self._token_embedding(contexts[FROM_TOKEN])
        encoded_to_tokens = self._token_embedding(contexts[TO_TOKEN])

        # [total paths; embeddings size]
        encoded_from_types = self._type_embedding(contexts[FROM_TYPE])
        encoded_to_types = self._type_embedding(contexts[TO_TYPE])

        # [total_paths; rnn size * num directions]
        encoded_paths = self._path_nodes_embedding(contexts[PATH_NODES])

        # [total_paths; output size]
        output = self._concat_with_linear(
            [encoded_from_types, encoded_from_tokens, encoded_paths, encoded_to_tokens, encoded_to_types]
        )
        return output
