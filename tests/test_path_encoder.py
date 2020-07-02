from os.path import join
from unittest import TestCase

from configs import EncoderConfig
from dataset import BufferedPathContext
from dataset.path_context_dataset import collate_path_contexts
from model.modules import PathEncoder
from tests.tools import get_path_to_test_data
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES


class TestPathEncoder(TestCase):

    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _hidden_size = 64

    def test_forward(self):
        config = EncoderConfig(self._hidden_size, self._hidden_size, True, 0.5, 0.5)

        buffered_path_contexts = BufferedPathContext.load(self._test_data_path)
        samples, true_labels, paths_for_label = collate_path_contexts(
            [buffered_path_contexts[i] for i in range(len(buffered_path_contexts))]
        )
        token_vocab_size = max(samples[FROM_TOKEN].max().item(), samples[TO_TOKEN].max().item())
        type_vocab_size = samples[PATH_TYPES].max().item()

        model = PathEncoder(config, self._hidden_size, token_vocab_size + 1, 0, type_vocab_size + 1, 0)

        out = model(samples)
        number_of_paths = sum(paths_for_label)
        self.assertTupleEqual((number_of_paths, self._hidden_size), out.shape)
