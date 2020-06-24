from unittest import TestCase

import numpy

from configs import PreprocessingConfig
from data_loaders import BufferedPathContext, Vocabulary
from utils.common import SOS, EOS, PAD, UNK


class TestBufferedPathContext(TestCase):
    def test_store_path_context(self):
        vocab = Vocabulary(
            token_to_id={SOS: 0, EOS: 1, PAD: 2, UNK: 3, "a": 4, "b": 5, "c": 6},
            type_to_id={SOS: 1, EOS: 2, PAD: 0, UNK: 3, "a": 4, "b": 5, "c": 6},
            label_to_id={SOS: 2, EOS: 0, PAD: 1, UNK: 3, "a": 4, "b": 5, "c": 6},
        )
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3, 2)
        buffered_path_context = BufferedPathContext(config, vocab)

        buffered_path_context.store_path_context(0, [4], [[4], [5, 6]], [[4, 5], [6]], [[6], [4, 5]])
        buffered_path_context.store_path_context(1, [], [[], []], [[], []], [[], []])
        buffered_path_context.store_path_context(
            2, [4, 5, 6], [[4, 5, 6], [6, 5, 4, 6]], [[6, 5, 4, 5], [4, 5, 6]], [[5, 5, 5], [4, 6, 4, 4]]
        )

        numpy.testing.assert_array_equal(
            buffered_path_context.labels_array, numpy.array([[2, 2, 2], [4, 0, 4], [0, 1, 5], [1, 1, 6]]),
        )

        numpy.testing.assert_array_equal(
            buffered_path_context.from_tokens_array,
            numpy.array(
                [[[0, 0], [0, 0], [0, 0]], [[4, 5], [1, 1], [4, 6]], [[1, 6], [2, 2], [5, 5]], [[2, 1], [2, 2], [6, 4]]]
            ),
        )

        numpy.testing.assert_array_equal(
            buffered_path_context.path_types_array,
            numpy.array(
                [[[1, 1], [1, 1], [1, 1]], [[4, 6], [2, 2], [6, 4]], [[5, 2], [0, 0], [5, 5]], [[2, 0], [0, 0], [4, 6]]]
            ),
        )

        numpy.testing.assert_array_equal(
            buffered_path_context.to_tokens_array,
            numpy.array(
                [[[0, 0], [0, 0], [0, 0]], [[6, 4], [1, 1], [5, 4]], [[1, 5], [2, 2], [5, 6]], [[2, 1], [2, 2], [5, 4]]]
            ),
        )

    def test_store_path_context_raise_error(self):
        vocab = Vocabulary(token_to_id={SOS: 0, PAD: 1}, type_to_id={SOS: 1, PAD: 2}, label_to_id={SOS: 2, PAD: 0})
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3, 2)
        buffered_path_context = BufferedPathContext(config, vocab)
        with self.assertRaises(ValueError):
            buffered_path_context.store_path_context(0, [], [[], []], [[]], [[], [], []])
