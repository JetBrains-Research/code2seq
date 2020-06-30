import pickle
from os.path import join
from unittest import TestCase

import torch

from configs import EncoderConfig
from dataset.path_context_dataset import collate_path_contexts
from model.modules import PathEncoder
from tests.tools import get_path_to_test_data
from utils.common import PATHS_FOR_LABEL


class TestPathEncoder(TestCase):

    _test_vocab_path = join(get_path_to_test_data(), "vocabulary.pkl")
    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _hidden_size = 64

    def test_forward(self):
        with open(self._test_vocab_path, "rb") as pkl_file:
            vocab = pickle.load(pkl_file)
        config = EncoderConfig(self._hidden_size, self._hidden_size, True, 0.5, 0.5)

        model = PathEncoder(vocab, config, self._hidden_size)

        with open(self._test_data_path, "rb") as pkl_file:
            buffered_path_contexts = pickle.load(pkl_file)

        samples, true_labels = collate_path_contexts(
            [buffered_path_contexts[i] for i in range(len(buffered_path_contexts))]
        )

        out = model(samples)
        number_of_paths = sum(samples[PATHS_FOR_LABEL])
        self.assertTupleEqual((number_of_paths, self._hidden_size), out[0].shape)
        torch.testing.assert_allclose(samples[PATHS_FOR_LABEL], out[1])
