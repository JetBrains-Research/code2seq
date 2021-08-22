import torch
from omegaconf import DictConfig
from torch import nn

from code2seq.model.modules import PathEncoder


class TypedPathEncoder(PathEncoder):
    def __init__(
        self,
        config: DictConfig,
        n_tokens: int,
        token_pad_id: int,
        n_nodes: int,
        node_pad_id: int,
        n_types: int,
        type_pad_id: int,
    ):
        super().__init__(config, n_tokens, token_pad_id, n_nodes, node_pad_id)

        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)

    @staticmethod
    def _calculate_concat_size(embedding_size: int, rnn_size: int, num_directions: int) -> int:
        return embedding_size * 4 + rnn_size * num_directions

    def _type_embedding(self, types: torch.Tensor) -> torch.Tensor:
        return self.type_embedding(types).sum(0)

    def forward(  # type: ignore
        self,
        from_type: torch.Tensor,
        from_token: torch.Tensor,
        path_nodes: torch.Tensor,
        to_token: torch.Tensor,
        to_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode each path context into the vector

        :param from_type: [n contexts; max type parts] types of start tokens
        :param from_token: [n contexts; max token parts] start tokens
        :param path_nodes: [n contexts; path nodes] path nodes
        :param to_token: [n contexts; max tokens parts] end tokens
        :param to_type: [n contexts; max types parts] types of end tokens
        :return: [n contexts; encoder size]
        """
        # [total paths; embedding size]
        encoded_from_tokens = self._token_embedding(from_token)
        encoded_to_tokens = self._token_embedding(to_token)

        # [total paths; embeddings size]
        encoded_from_types = self._type_embedding(from_type)
        encoded_to_types = self._type_embedding(to_type)

        # [total_paths; rnn size * num directions]
        encoded_paths = self._path_nodes_embedding(path_nodes)

        # [total_paths; output size]
        output = self._concat_with_linear(
            [encoded_from_types, encoded_from_tokens, encoded_paths, encoded_to_tokens, encoded_to_types]
        )
        return output
