import pickle
from collections import Counter
from multiprocessing import cpu_count
from os import path
from unittest import TestCase

import numpy

from configs import get_preprocessing_config_code2seq_params
from data_preprocessing.preprocess_astminer_code2vec_data import preprocess_csv, convert_path_context_to_ids
from tests.tools import get_path_to_test_data
from utils.preprocessing import convert_holdout, vocab_from_counters


class TestConvertHoldout(TestCase):
    _test_data_path = path.join(get_path_to_test_data(), "poj_104-test")

    def test_preprocess_csv(self):
        tokens, node_types, paths = preprocess_csv(self._test_data_path, "test")

        self.assertDictEqual(tokens, {1: ["int"], 2: ["sum"], 3: ["m"], 4: ["f"], 5: ["n", "d"], 6: ["i"]})

        self.assertDictEqual(
            node_types, {1: "TYPE_FULL_NAME", 2: "LOCAL", 3: "LOCAL", 4: "NAME", 5: "NAME", 6: "TYPE_FULL_NAME",},
        )
        self.assertDictEqual(paths, {1: [1, 2, 3, 4], 2: [5, 2, 3, 6]})

    def test_convert_holdout(self):
        tokens, node_types, paths = preprocess_csv(self._test_data_path, "test")
        config = get_preprocessing_config_code2seq_params("poj_104-test")

        token_counter = Counter({"int": 1, "sum": 2, "m": 3, "f": 4, "n": 5, "d": 6, "i": 7})
        type_counter = Counter({"LOCAL": 1, "TYPE_FULL_NAME": 2, "NAME": 3,})
        target_counter = Counter({"1": 1})
        vocab = vocab_from_counters(config, token_counter, target_counter, type_counter)

        convert_holdout(
            self._test_data_path,
            "test",
            vocab,
            config,
            cpu_count(),
            convert_path_context_to_ids,
            paths=paths,
            tokens=tokens,
            node_types=node_types,
        )
        holdout_output_folder = path.join(self._test_data_path, "test")
        with open(path.join(holdout_output_folder, "buffered_paths_0.pkl"), "rb") as buf_paths:
            buf, labels, contexts_per_label = pickle.load(buf_paths)
            from_tokens = numpy.array(
                [[10, 8, 7, 10, 10], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
            )
            to_tokens = numpy.array(
                [[9, 10, 10, 6, 4], [2, 2, 2, 5, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
            )
            path_types = numpy.array(
                [
                    [0, 0, 0, 0, 0],
                    [5, 4, 4, 5, 5],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [4, 5, 5, 4, 4],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                ]
            )
            numpy.testing.assert_array_equal(buf["path_types"], path_types)
            numpy.testing.assert_array_equal(buf["from_token"], from_tokens)
            numpy.testing.assert_array_equal(buf["to_token"], to_tokens)
            numpy.testing.assert_array_equal(contexts_per_label, [5])
            numpy.testing.assert_array_equal(labels, numpy.array([[0], [4], [1], [2], [2], [2], [2]]))
