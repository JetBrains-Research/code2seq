from os.path import join
from unittest import TestCase

import torch

from configs import DecoderConfig
from dataset import BufferedPathContext
from dataset.path_context_dataset import PathContextBatch
from model.modules import PathDecoder
from tests.tools import get_path_to_test_data


class TestPathDecoder(TestCase):

    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _hidden_size = 64
    _target_length = 10
    _out_size = 128

    def test_forward(self):
        config = DecoderConfig(self._hidden_size, self._hidden_size, 1, 0.5, 1)

        model = PathDecoder(config, self._out_size, 0, 0)

        buffered_path_contexts = BufferedPathContext.load(self._test_data_path)

        batch = PathContextBatch([buffered_path_contexts[i] for i in range(len(buffered_path_contexts))])
        number_of_paths = sum(batch.contexts_per_label)
        fake_encoder_input = torch.rand(number_of_paths, self._hidden_size)

        output = model(fake_encoder_input, batch.contexts_per_label, self._target_length)

        self.assertTupleEqual((self._target_length, len(batch.contexts_per_label), self._out_size), output.shape)
