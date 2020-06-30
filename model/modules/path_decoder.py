import torch
from torch import nn

from configs import DecoderConfig


class PathDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, encoder_size: int, out_size: int, max_output_length: int):
        super().__init__()
        self.max_output_length = max_output_length
        self.beam_width = config.beam_width

        self.decoder_lstm = nn.LSTM(encoder_size, config.decoder_size, num_layers=config.num_decoder_layers)
        self.attention = nn.Linear(config.decoder_size, config.decoder_size)
        self.projection_layer = nn.Linear(config.decoder_size, out_size)

    def forward(self, encoded_paths: torch.Tensor, paths_for_label: torch.LongTensor):
        pass
