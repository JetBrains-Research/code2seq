from os.path import join
from unittest import TestCase

import torch

from configs import Code2SeqTestConfig
from dataset import PathContextDataset, PathContextBatch
from model.modules import PathDecoder
from utils.common import VOCABULARY_NAME, TRAIN_HOLDOUT
from utils.filesystem import get_path_to_test_data
from utils.vocabulary import Vocabulary


class TestPathDecoder(TestCase):

    _train_path = join(get_path_to_test_data(), f"java-test.{TRAIN_HOLDOUT}.c2s")
    _vocabulary_path = join(get_path_to_test_data(), VOCABULARY_NAME)

    def test_forward(self):
        config = Code2SeqTestConfig()

        vocabulary = Vocabulary.load_vocabulary(self._vocabulary_path)
        dataset = PathContextDataset(
            self._train_path, vocabulary, config.data_processing, config.hyper_parameters.max_context, False
        )

        model = PathDecoder(config.decoder_config, len(vocabulary.label_to_id), 0, 0)

        batch = PathContextBatch([dataset[i] for i in range(config.hyper_parameters.batch_size)])
        number_of_paths = sum(batch.contexts_per_label)

        fake_encoder_output = torch.rand(number_of_paths, config.decoder_config.decoder_size)
        output = model(fake_encoder_output, batch.contexts_per_label, config.data_processing.max_target_parts)

        true_shape = (
            config.data_processing.max_target_parts,
            config.hyper_parameters.batch_size,
            len(vocabulary.label_to_id),
        )
        self.assertTupleEqual(true_shape, output.shape)
