import math

import torch
from commode_utils.training import cut_into_segments
from omegaconf import DictConfig
from torch import nn, Tensor, LongTensor
from torch.nn import Embedding, Linear
from torch.nn.modules.transformer import TransformerDecoderLayer, Transformer, TransformerDecoder
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_token_length: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_token_length).reshape(max_token_length, 1)

        pe = torch.zeros((max_token_length, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, token_embedding: Tensor):
        output = token_embedding + self.pe[:, : token_embedding.size(1), :]
        return self.dropout(output)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerCommentDecoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        vocab_size: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._pad_token = pad_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self._teacher_forcing = teacher_forcing

        self._embedding = TokenEmbedding(vocab_size, config.decoder_size)
        self._positional_encoding = PositionalEncoding(config.decoder_size, config.decoder_dropout)
        decoder_layer = TransformerDecoderLayer(
            d_model=config.decoder_size,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.decoder_dropout,
            batch_first=True,
        )
        self._decoder = TransformerDecoder(decoder_layer, config.decoder_num_layers)
        self._linear = Linear(config.decoder_size, vocab_size)

    def decode(
        self, target_sequence: Tensor, batched_encoder_output: Tensor, tgt_mask: Tensor, attention_mask: Tensor
    ) -> Tensor:
        tgt_key_padding_mask = target_sequence == self._pad_token

        embedded = self._embedding(target_sequence)
        positionally_encoded = self._positional_encoding(embedded)
        decoded = self._decoder(
            tgt=positionally_encoded,
            memory=batched_encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=attention_mask,
        )
        return self._linear(decoded)

    def forward(
        self,
        encoder_output: Tensor,
        segment_sizes: LongTensor,
        output_size: int,
        target_sequence: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        device = encoder_output.get_device()
        batch_size = segment_sizes.shape[0]

        batched_encoder_output, attention_mask = cut_into_segments(encoder_output, segment_sizes)
        # TODO fill attentions with smth good
        attentions = batched_encoder_output.new_zeros((output_size, batch_size, attention_mask.shape[1]))

        if target_sequence is not None:
            target_sequence = target_sequence.permute(1, 0)

            tgt_mask = (Transformer.generate_square_subsequent_mask(output_size)).to(device)

            output = self.decode(target_sequence, batched_encoder_output, tgt_mask, attention_mask)
        else:
            with torch.no_grad():
                output = torch.zeros((batch_size, output_size, self._vocab_size)).to(device)

                target_sequence = torch.zeros((batch_size, 1)).to(device)
                target_sequence[:, 0] = self._sos_token
                is_ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

                for i in range(output_size):
                    tgt_mask = (Transformer.generate_square_subsequent_mask(i + 1)).to(device)
                    logits = self.decode(target_sequence, batched_encoder_output, tgt_mask, attention_mask)

                    prediction = logits.argmax(-1)[:, i]
                    target_sequence = torch.cat((target_sequence, prediction.unsqueeze(1)), dim=1)
                    output[:, i, :] = logits[:, i, :]

                    is_ended = torch.logical_or(is_ended, (prediction == self._eos_token))
                    if torch.count_nonzero(is_ended).item() == batch_size:
                        break

        return output.permute(1, 0, 2), attentions
