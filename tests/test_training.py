from unittest import TestCase

import torch

from code2seq.utils.training import cut_encoded_contexts


class TestTrainingUtils(TestCase):
    def test_cut_encoded_contexts(self):
        units = 10
        mask_value = -1
        batch_size = 5
        contexts_per_label = list(range(1, batch_size + 1))
        max_context_len = max(contexts_per_label)

        encoded_contexts = torch.cat([torch.full((i, units), i, dtype=torch.float) for i in contexts_per_label])

        def create_true_batch(fill_value: int, counts: int, size: int) -> torch.tensor:
            return torch.cat(
                [torch.full((1, counts, units), fill_value, dtype=torch.float), torch.zeros((1, size - counts, units))],
                dim=1,
            )

        def create_batch_mask(counts: int, size: int) -> torch.tensor:
            return torch.cat(
                [torch.zeros(1, counts), torch.full((1, size - counts), mask_value, dtype=torch.float)], dim=1
            )

        true_batched_context = torch.cat([create_true_batch(i, i, max_context_len) for i in contexts_per_label])
        true_attention_mask = torch.cat([create_batch_mask(i, max_context_len) for i in contexts_per_label])

        batched_context, attention_mask = cut_encoded_contexts(encoded_contexts, contexts_per_label, mask_value)

        torch.testing.assert_allclose(batched_context, true_batched_context)
        torch.testing.assert_allclose(attention_mask, true_attention_mask)
