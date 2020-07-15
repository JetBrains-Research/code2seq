import torch
import torch.nn.functional as F
from torch import nn


class LuongAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, units, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate attention weights

        :param hidden: [batch size; units]
        :param encoder_outputs: [batch size; seq len; units]
        :param mask: [batch size; seq len]
        :return: [batch size; 1; seq len]
        """
        # [batch size; 1; units]
        hidden = hidden.unsqueeze(1)
        # [batch size; units; seq len]
        encoder_outputs = encoder_outputs.transpose(1, 2)
        # [batch size; 1; units]
        score = self.attn(hidden)
        # [batch size; 1; seq len]
        score = torch.bmm(score, encoder_outputs)
        score += mask.unsqueeze(1)

        # [batch size; 1; seq len]
        weights = F.softmax(score, dim=-1)
        return weights
