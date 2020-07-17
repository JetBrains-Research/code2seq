from unittest import TestCase
import pickle
from os import path
from multiprocessing import cpu_count
import numpy as np
from collections import Counter

from data_preprocessing.buffer_utils import convert_holdout
from data_preprocessing.preprocess_astminer_code2vec_data import preprocess_csv, convert_path_context_to_ids
from configs import get_preprocessing_config_code2seq_params
from utils.common import vocab_from_counters
from tests.tools import get_path_to_test_data


class TestConvertHoldout(TestCase):
    def test_preprocess_csv(self):
        _test_data_path = get_path_to_test_data("poj_104-test")
        preprocess_csv(_test_data_path, "test")

        test_parsed_path = path.join(_test_data_path, "test.pkl")
        with open(test_parsed_path, "rb") as train_parsed_data:
            data = pickle.load(train_parsed_data)
            tokens, node_types, paths = data["tokens"], data["node_types"], data["paths"]

        self.assertEqual(tokens, {1: ["int"], 2: ["sum"], 3: ["m"], 4: ["f"], 5: ["n", "d"], 6: ["i"]})

        self.assertEqual(
            node_types,
            {
                1: "TYPE_FULL_NAME UP",
                2: "LOCAL UP",
                3: "LOCAL DOWN",
                4: "NAME DOWN",
                5: "NAME UP",
                6: "TYPE_FULL_NAME DOWN",
            },
        )
        self.assertEqual(paths, {1: [1, 2, 3, 4], 2: [5, 2, 3, 6]})

    def test_convert_holdout(self):
        _test_data_path = get_path_to_test_data("poj_104-test")
        preprocess_csv(_test_data_path, "test")
        config = get_preprocessing_config_code2seq_params("poj_104-test")

        token_counter = Counter({"int": 1, "sum": 2, "m": 3, "f": 4, "n": 5, "d": 6, "i": 7})
        type_counter = Counter(
            {
                "LOCAL UP": 1,
                "LOCAL DOWN": 2,
                "TYPE_FULL_NAME UP": 3,
                "NAME DOWN": 4,
                "NAME UP": 5,
                "TYPE_FULL_NAME DOWN": 6,
            }
        )
        target_counter = Counter({"1": 1})

        vocab = vocab_from_counters(config, token_counter, target_counter, type_counter)

        holdout_data_path = path.join(_test_data_path, f"path_contexts.test.csv")
        holdout_output_folder = path.join(_test_data_path, "test")
        holdout_parsed_path = path.join(_test_data_path, "test.pkl")
        with open(holdout_parsed_path, "rb") as parsed_data:
            data = pickle.load(parsed_data)
            tokens, node_types, paths = data["tokens"], data["node_types"], data["paths"]

        convert_holdout(
            holdout_data_path,
            holdout_output_folder,
            vocab,
            config,
            cpu_count(),
            convert_path_context_to_ids,
            paths=paths,
            tokens=tokens,
            node_types=node_types,
        )

        with open(path.join(holdout_output_folder, "buffered_paths_0.pkl"), "rb") as buf_paths:
            buf, labels, contexts_per_label = pickle.load(buf_paths)
            from_tokens = np.array(
                [[10, 8, 7, 10, 10], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
            )
            to_tokens = np.array(
                [[9, 10, 10, 6, 4], [2, 2, 2, 5, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]
            )
            path_types = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [7, 5, 5, 7, 7],
                    [9, 9, 9, 9, 9],
                    [8, 8, 8, 8, 8],
                    [6, 4, 4, 6, 6],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2],
                ]
            )
            np.testing.assert_array_equal(buf["path_types"], path_types)
            np.testing.assert_array_equal(buf["from_token"], from_tokens)
            np.testing.assert_array_equal(buf["to_token"], to_tokens)
            np.testing.assert_array_equal(contexts_per_label, [5])
            np.testing.assert_array_equal(labels, np.array([[0], [4], [1], [2], [2], [2], [2]]))
