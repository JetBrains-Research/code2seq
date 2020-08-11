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
        :return: [batch size; seq len]
        """
        batch_size, seq_len = mask.shape
        # [batch size; units]
        attended_hidden = self.attn(hidden)
        # [batch size; seq len]
        score = torch.bmm(encoder_outputs, attended_hidden.view(batch_size, -1, 1)).squeeze(-1)
        score += mask

        # [batch size; seq len]
        weights = F.softmax(score, dim=1)
        return weights


class LocalAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate attention weights

        :param encoder_outputs: [batch size; seq len; units]
        :param mask: [batch size; seq len]
        :return: [batch size; 1; seq len]
        """
        # [batch size; seq len; 1]
        attended_encoder_outputs = self.attn(encoder_outputs)
        attended_encoder_outputs += mask.unsqueeze(2)

        # [batch size; seq len; 1]
        weights = torch.softmax(attended_encoder_outputs, dim=1)

        return weights
