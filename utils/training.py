from typing import List, Tuple

import torch

from utils.common import segment_sizes_to_slices


def cut_encoded_contexts(
    encoded_contexts: torch.Tensor, contexts_per_label: List[int], mask_value: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut encoded contexts into batches

    :param encoded_contexts: [n contexts; units]
    :param contexts_per_label: [batch size]
    :param mask_value:
    :return: [batch size; max context len; units], [batch size; max context len]
    """
    batch_size = len(contexts_per_label)
    max_context_len = max(contexts_per_label)

    batched_contexts = encoded_contexts.new_zeros((batch_size, max_context_len, encoded_contexts.shape[-1]))
    attention_mask = encoded_contexts.new_zeros((batch_size, max_context_len))

    context_slices = segment_sizes_to_slices(contexts_per_label)
    for i, (cur_slice, cur_size) in enumerate(zip(context_slices, contexts_per_label)):
        batched_contexts[i, :cur_size] = encoded_contexts[cur_slice]
        attention_mask[i, cur_size:] = mask_value

    return batched_contexts, attention_mask
