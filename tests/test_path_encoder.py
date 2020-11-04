from os.path import join
from unittest import TestCase

from configs import Code2SeqTestConfig
from dataset import PathContextDataset, PathContextBatch
from model.modules import PathEncoder
from utils.common import VOCABULARY_NAME, PAD, TRAIN_HOLDOUT
from utils.filesystem import get_path_to_test_data
from utils.vocabulary import Vocabulary


class TestPathEncoder(TestCase):

    _train_path = join(get_path_to_test_data(), f"java-test.{TRAIN_HOLDOUT}.c2s")
    _vocabulary_path = join(get_path_to_test_data(), VOCABULARY_NAME)

    def test_forward(self):
        config = Code2SeqTestConfig()

        vocabulary = Vocabulary.load_vocabulary(self._vocabulary_path)
        dataset = PathContextDataset(
            self._train_path, vocabulary, config.data_processing, config.hyper_parameters.max_context, False
        )

        batch = PathContextBatch([dataset[i] for i in range(config.hyper_parameters.batch_size)])

        model = PathEncoder(
            config.encoder_config,
            config.decoder_config.decoder_size,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[PAD],
            len(vocabulary.type_to_id),
            vocabulary.type_to_id[PAD],
        )
        output = model(batch.contexts)

        true_shape = (sum(batch.contexts_per_label), config.decoder_config.decoder_size)
        self.assertTupleEqual(true_shape, output.shape)
