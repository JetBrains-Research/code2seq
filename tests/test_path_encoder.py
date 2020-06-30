import pickle
from os.path import join
from unittest import TestCase

from configs import EncoderConfig
from dataset.path_context_dataset import collate_path_contexts
from model.modules import PathEncoder
from tests.tools import get_path_to_test_data


class TestPathEncoder(TestCase):

    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _vocab_size = 128
    _hidden_size = 64

    def test_forward(self):
        config = EncoderConfig(self._hidden_size, self._hidden_size, True, 0.5, 0.5)

        model = PathEncoder(config, self._hidden_size, self._vocab_size, 0, self._vocab_size, 0)

        with open(self._test_data_path, "rb") as pkl_file:
            buffered_path_contexts = pickle.load(pkl_file)

        samples, true_labels, paths_for_label = collate_path_contexts(
            [buffered_path_contexts[i] for i in range(len(buffered_path_contexts))]
        )

        out = model(samples)
        number_of_paths = sum(paths_for_label)
        self.assertTupleEqual((number_of_paths, self._hidden_size), out.shape)
