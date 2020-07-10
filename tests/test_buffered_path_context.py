from unittest import TestCase

import numpy

from configs import PreprocessingConfig
from dataset import Vocabulary, create_standard_bpc
from dataset.buffered_path_context import _prepare_to_store
from utils.common import SOS, EOS, PAD, FROM_TOKEN, PATH_TYPES, TO_TOKEN


class TestBufferedPathContext(TestCase):
    def test_creating_standard_path_context(self):
        vocab = Vocabulary(
            token_to_id={SOS: 0, EOS: 1, PAD: 2},
            type_to_id={SOS: 1, EOS: 2, PAD: 0},
            label_to_id={SOS: 2, EOS: 0, PAD: 1},
        )
        config = PreprocessingConfig("", 3, 3, 3, True, True, True, -1, -1, 3)
        labels = [[4], [], [4, 5, 6]]
        from_tokens = [
            [[4], [5, 6]],
            [[], [], []],
            [[6, 5, 4]],
        ]
        path_types = [
            [[4, 5], [6]],
            [[], [], []],
            [[6, 5, 4]],
        ]
        to_tokens = [
            [[6], [4, 5]],
            [[], [], []],
            [[4, 6, 4]],
        ]

        buffered_path_context = create_standard_bpc(config, vocab, labels, from_tokens, path_types, to_tokens)

        true_labels = numpy.array([[2, 2, 2], [4, 0, 4], [0, 1, 5], [1, 1, 6]])
        true_from_tokens = numpy.array([[0, 0, 0, 0, 0, 0], [4, 5, 1, 1, 1, 6], [1, 6, 2, 2, 2, 5], [2, 1, 2, 2, 2, 4]])
        true_path_types = numpy.array([[1, 1, 1, 1, 1, 1], [4, 6, 2, 2, 2, 6], [5, 2, 0, 0, 0, 5], [2, 0, 0, 0, 0, 4]])
        true_to_tokens = numpy.array([[0, 0, 0, 0, 0, 0], [6, 4, 1, 1, 1, 4], [1, 5, 2, 2, 2, 6], [2, 1, 2, 2, 2, 4]])

        self.assertListEqual([2, 3, 1], buffered_path_context.contexts_per_label)
        numpy.testing.assert_array_equal(true_labels, buffered_path_context.labels)
        numpy.testing.assert_array_equal(true_from_tokens, buffered_path_context.contexts[FROM_TOKEN])
        numpy.testing.assert_array_equal(true_path_types, buffered_path_context.contexts[PATH_TYPES])
        numpy.testing.assert_array_equal(true_to_tokens, buffered_path_context.contexts[TO_TOKEN])

    def test_creating_standard_path_context_check_path_shapes(self):
        config = PreprocessingConfig("", 3, 3, 3, True, True, True, -1, -1, 3)
        with self.assertRaises(ValueError):
            create_standard_bpc(config, Vocabulary(), [[]], [[], []], [[], [], []], [[]])

    def test_creating_standard_path_context_check_full_buffer(self):
        config = PreprocessingConfig("", 3, 3, 3, True, True, True, -1, -1, 3)
        with self.assertRaises(ValueError):
            create_standard_bpc(config, Vocabulary(), [[], [], []], [[], []], [[], [], []], [[]])

    def test__prepare_to_store_simple(self):
        values = [3, 4, 5]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 1, 2]
        self.assertListEqual(true_result, _prepare_to_store(values, 5, to_id, True))

    def test__prepare_to_store_long(self):
        values = [3, 4, 5, 6, 7, 8, 9, 10]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 6, 7]
        self.assertListEqual(true_result, _prepare_to_store(values, 5, to_id, True))

    def test__prepare_to_store_short(self):
        values = [3]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 1, 2, 2, 2]
        self.assertListEqual(true_result, _prepare_to_store(values, 5, to_id, True))

    def test__prepare_to_store_no_wrap(self):
        values = [3, 4, 5]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [3, 4, 5, 2, 2]
        self.assertListEqual(true_result, _prepare_to_store(values, 5, to_id, False))
