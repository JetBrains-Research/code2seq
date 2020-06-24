from typing import List, Any
from unittest import TestCase

import numpy

from configs import PreprocessingConfig
from data_loaders import BufferedPathContext, Vocabulary
from utils.common import SOS, EOS, PAD, UNK


class TestBufferedPathContext(TestCase):
    @staticmethod
    def _create_test_buffered_path_context() -> BufferedPathContext:
        vocab = Vocabulary(
            token_to_id={SOS: 0, EOS: 1, PAD: 2},
            type_to_id={SOS: 1, EOS: 2, PAD: 0},
            label_to_id={SOS: 2, EOS: 0, PAD: 1},
        )
        config = PreprocessingConfig("", 3, 3, 3, -1, -1, 3)
        return BufferedPathContext(config, vocab)

    def _assert_list_of_lists_equal(self, expected_list: List[List[Any]], actual_list: List[List[Any]]):
        self.assertEqual(len(expected_list), len(actual_list), "Unequal lengths of lists")
        for pos, (_exp_l, _act_l) in enumerate(zip(expected_list, actual_list)):
            self.assertListEqual(_exp_l, _act_l, f"Found unequal sub lists on {pos} position")

    def test_store_path_context(self):
        buffered_path_context = self._create_test_buffered_path_context()

        buffered_path_context.store_path_context([4], [[4], [5, 6]], [[4, 5], [6]], [[6], [4, 5]])
        buffered_path_context.store_path_context([], [[], [], []], [[], [], []], [[], [], []])
        buffered_path_context.store_path_context([4, 5, 6], [[6, 5, 4, 6]], [[6, 5, 4, 5]], [[4, 6, 4, 4]])

        true_labels_array = [[2, 4, 0, 1], [2, 0, 1, 1], [2, 4, 5, 6]]
        true_from_tokens_array = [
            [[0, 4, 1, 2], [0, 5, 6, 1]],
            [[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]],
            [[0, 6, 5, 4]],
        ]
        true_path_types_array = [
            [[1, 4, 5, 2], [1, 6, 2, 0]],
            [[1, 2, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            [[1, 6, 5, 4]],
        ]
        true_to_tokens_array = [
            [[0, 6, 1, 2], [0, 4, 5, 1]],
            [[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]],
            [[0, 4, 6, 4]],
        ]

        self._assert_list_of_lists_equal(true_labels_array, buffered_path_context.labels_array)
        self._assert_list_of_lists_equal(true_from_tokens_array, buffered_path_context.from_tokens_array)
        self._assert_list_of_lists_equal(true_path_types_array, buffered_path_context.path_types_array)
        self._assert_list_of_lists_equal(true_to_tokens_array, buffered_path_context.to_tokens_array)

    def test_store_path_context_check_path_shapes(self):
        buffered_path_context = self._create_test_buffered_path_context()
        with self.assertRaises(ValueError):
            buffered_path_context.store_path_context([], [[], []], [[]], [[], [], []])

    def test_store_path_context_check_full_buffer(self):
        buffered_path_context = self._create_test_buffered_path_context()
        for _ in range(3):
            buffered_path_context.store_path_context([], [[]], [[]], [[]])
        with self.assertRaises(RuntimeError):
            buffered_path_context.store_path_context([], [[], []], [[]], [[], [], []])

    def test__prepare_to_store_simple(self):
        values = [3, 4, 5]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 1, 2]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))

    def test__prepare_to_store_long(self):
        values = [3, 4, 5, 6, 7, 8, 9, 10]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 4, 5, 6, 7]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))

    def test__prepare_to_store_short(self):
        values = [3]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = [0, 3, 1, 2, 2, 2]
        self.assertListEqual(true_result, BufferedPathContext._prepare_to_store(values, 5, to_id))
