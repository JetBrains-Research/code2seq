import pickle
from os.path import join
from unittest import TestCase

import torch

from configs import DecoderConfig
from dataset.path_context_dataset import collate_path_contexts
from model.modules import PathDecoder
from tests.tools import get_path_to_test_data
from utils.common import PATHS_FOR_LABEL


class TestPathDecoder(TestCase):

    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _hidden_size = 64
    _target_length = 10
    _out_size = 128

    def test_forward(self):
        config = DecoderConfig(self._hidden_size, 1)

        model = PathDecoder(config, self._hidden_size, self._out_size, self._target_length, 0, 0)

        with open(self._test_data_path, "rb") as pkl_file:
            buffered_path_contexts = pickle.load(pkl_file)

        samples, true_labels, paths_for_labels = collate_path_contexts(
            [buffered_path_contexts[i] for i in range(len(buffered_path_contexts))]
        )
        number_of_paths = sum(paths_for_labels)
        fake_encoder_input = torch.rand(number_of_paths, self._hidden_size)

        output = model(fake_encoder_input, paths_for_labels)

        self.assertTupleEqual((self._target_length, len(paths_for_labels), self._out_size), output.shape)
