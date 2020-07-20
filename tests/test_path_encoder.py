from os.path import join
from unittest import TestCase

from configs import EncoderConfig
from dataset import BufferedPathContext
from dataset.path_context_dataset import PathContextBatch
from model.modules import PathEncoder
from tests.tools import get_path_to_test_data
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES


class TestPathEncoder(TestCase):

    _test_data_path = join(get_path_to_test_data(), "train", "buffered_paths_0.pkl")
    _hidden_size = 64
    _batch_size = 128

    def test_forward(self):
        config = EncoderConfig(self._hidden_size, self._hidden_size, True, 0.5, 1, 0.5)

        buffered_path_contexts = BufferedPathContext.load(self._test_data_path)
        batch = PathContextBatch([buffered_path_contexts[i] for i in range(self._batch_size)])
        token_vocab_size = max(batch.context[FROM_TOKEN].max().item(), batch.context[TO_TOKEN].max().item())
        type_vocab_size = batch.context[PATH_TYPES].max().item()

        model = PathEncoder(config, self._hidden_size, token_vocab_size + 1, 0, type_vocab_size + 1, 0)

        out = model(batch.context)
        number_of_paths = sum(batch.contexts_per_label)
        self.assertTupleEqual((number_of_paths, self._hidden_size), out.shape)
