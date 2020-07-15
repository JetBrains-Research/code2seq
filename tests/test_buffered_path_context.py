from unittest import TestCase

import numpy

from dataset import BufferedPathContext, ConvertParameters
from utils.common import SOS, EOS, PAD, FROM_TOKEN, PATH_TYPES, TO_TOKEN


class TestBufferedPathContext(TestCase):
    def test_creating_standard_path_context(self):
        token_to_id = {SOS: 0, EOS: 1, PAD: 2}
        type_to_id = {SOS: 1, EOS: 2, PAD: 0}
        label_to_id = {SOS: 2, EOS: 0, PAD: 1}
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

        buffered_path_context = BufferedPathContext.create_from_lists(
            (labels, ConvertParameters(3, True, label_to_id)),
            {
                FROM_TOKEN: (from_tokens, ConvertParameters(3, False, token_to_id)),
                PATH_TYPES: (path_types, ConvertParameters(3, True, type_to_id)),
                TO_TOKEN: (to_tokens, ConvertParameters(3, False, token_to_id)),
            },
        )

        true_labels = numpy.array([[2, 2, 2], [4, 0, 4], [0, 1, 5], [1, 1, 6]])
        true_from_tokens = numpy.array([[4, 5, 2, 2, 2, 6], [2, 6, 2, 2, 2, 5], [2, 2, 2, 2, 2, 4]])
        true_path_types = numpy.array([[1, 1, 1, 1, 1, 1], [4, 6, 2, 2, 2, 6], [5, 2, 0, 0, 0, 5], [2, 0, 0, 0, 0, 4]])
        true_to_tokens = numpy.array([[6, 4, 2, 2, 2, 4], [2, 5, 2, 2, 2, 6], [2, 2, 2, 2, 2, 4]])

        self.assertListEqual([2, 3, 1], buffered_path_context.contexts_per_label)
        numpy.testing.assert_array_equal(true_labels, buffered_path_context.labels)
        numpy.testing.assert_array_equal(true_from_tokens, buffered_path_context.contexts[FROM_TOKEN])
        numpy.testing.assert_array_equal(true_path_types, buffered_path_context.contexts[PATH_TYPES])
        numpy.testing.assert_array_equal(true_to_tokens, buffered_path_context.contexts[TO_TOKEN])

    def test_creating_buffered_context_check_context_shapes(self):
        with self.assertRaises(ValueError):
            BufferedPathContext.create_from_lists(
                ([[]], ConvertParameters(0, False, {})),
                {
                    FROM_TOKEN: ([[], []], ConvertParameters(0, False, {})),
                    PATH_TYPES: ([[], [], []], ConvertParameters(0, False, {})),
                    TO_TOKEN: ([[]], ConvertParameters(0, False, {})),
                },
            )

    def test_creating_buffered_context_check_labels_shape(self):
        with self.assertRaises(ValueError):
            BufferedPathContext.create_from_lists(
                ([[], [], []], ConvertParameters(0, False, {})),
                {
                    FROM_TOKEN: ([[], []], ConvertParameters(0, False, {})),
                    PATH_TYPES: ([[], []], ConvertParameters(0, False, {})),
                    TO_TOKEN: ([[], []], ConvertParameters(0, False, {})),
                },
            )

    def test__convert_list_to_numpy_array_simple(self):
        values = [[3, 4, 5]]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = numpy.array([[0], [3], [4], [5], [1], [2]])
        numpy.testing.assert_equal(
            true_result, BufferedPathContext._list_to_numpy_array(values, len(values), 5, True, to_id)
        )

    def test__convert_list_to_numpy_array_long(self):
        values = [[3, 4, 5, 6, 7, 8, 9, 10]]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = numpy.array([[0], [3], [4], [5], [6], [7]])
        numpy.testing.assert_equal(
            true_result, BufferedPathContext._list_to_numpy_array(values, len(values), 5, True, to_id)
        )

    def test__convert_list_to_numpy_array_short(self):
        values = [[3]]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = numpy.array([[0], [3], [1], [2], [2], [2]])
        numpy.testing.assert_equal(
            true_result, BufferedPathContext._list_to_numpy_array(values, len(values), 5, True, to_id)
        )

    def test__convert_list_to_numpy_no_wrap(self):
        values = [[3, 4, 5]]
        to_id = {SOS: 0, EOS: 1, PAD: 2}
        true_result = numpy.array([[3], [4], [5], [2], [2]])
        numpy.testing.assert_equal(
            true_result, BufferedPathContext._list_to_numpy_array(values, len(values), 5, False, to_id)
        )
