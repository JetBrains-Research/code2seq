from typing import Optional

import torch
from omegaconf import DictConfig

from code2seq.data.path_context import BatchedLabeledTypedPathContext
from code2seq.data.vocabulary import TypedVocabulary
from code2seq.model import Code2Seq
from code2seq.model.modules import TypedPathEncoder, PathEncoder


class TypedCode2Seq(Code2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: TypedVocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)
        self._vocabulary: TypedVocabulary = vocabulary

    def _get_encoder(self, config: DictConfig) -> PathEncoder:
        return TypedPathEncoder(
            config,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[TypedVocabulary.PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[TypedVocabulary.PAD],
            len(self._vocabulary.type_to_id),
            self._vocabulary.type_to_id[TypedVocabulary.PAD],
        )

    def forward(  # type: ignore
        self,
        from_type: torch.Tensor,
        from_token: torch.Tensor,
        path_nodes: torch.Tensor,
        to_token: torch.Tensor,
        to_type: torch.Tensor,
        contexts_per_label: torch.Tensor,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        encoded_paths = self._encoder(from_type, from_token, path_nodes, to_token, to_type)
        output_logits = self._decoder(encoded_paths, contexts_per_label, output_length, target_sequence)
        return output_logits

    def logits_from_batch(  # type: ignore[override]
        self, batch: BatchedLabeledTypedPathContext, target_sequence: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self(
            batch.from_type,
            batch.from_token,
            batch.path_nodes,
            batch.to_token,
            batch.to_type,
            batch.contexts_per_label,
            batch.labels.shape[0],
            target_sequence,
        )
