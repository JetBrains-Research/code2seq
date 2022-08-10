from commode_utils.training import cut_into_segments
from omegaconf import DictConfig
from torch import nn, Tensor, LongTensor
from torch.nn import TransformerDecoder, Embedding, Linear
from torch.nn.modules.transformer import TransformerDecoderLayer, Transformer
from typing import Tuple


class CommentDecoder(nn.Module):
    def __init__(
        self, config: DictConfig, vocab_size: int, pad_token: int, sos_token: int, teacher_forcing: float = 0.0
    ):
        super().__init__()
        self._pad_token = pad_token
        self._sos_token = sos_token
        self._teacher_forcing = teacher_forcing

        self._embedding = Embedding(vocab_size, config.decoder_size, padding_idx=pad_token)
        decoder_layer = TransformerDecoderLayer(
            d_model=config.decoder_size,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.decoder_dropout,
            batch_first=True,
        )
        self._decoder = TransformerDecoder(decoder_layer, config.decoder_num_layers)
        self._linear = Linear(config.decoder_size, vocab_size)

    def forward(
        self,
        encoder_output: Tensor,
        segment_sizes: LongTensor,
        output_size: int,
        target_sequence: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size = segment_sizes.shape[0]

        if not self.training:
            target_sequence = encoder_output.new_zeros((batch_size, output_size), dtype=int)
        else:
            target_sequence = target_sequence.permute(1, 0)

        batched_encoder_output, attention_mask = cut_into_segments(encoder_output, segment_sizes)
        attentions = batched_encoder_output.new_zeros((output_size, batch_size, attention_mask.shape[1]))

        embedded = self._embedding(target_sequence)

        tgt_mask = Transformer.generate_square_subsequent_mask(output_size).to(target_sequence.get_device())
        tgt_key_padding_mask = target_sequence == self._pad_token

        decoded = self._decoder(
            tgt=embedded, memory=batched_encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self._linear(decoded).permute(1, 0, 2)
        return output, attentions
