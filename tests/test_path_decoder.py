from os.path import join
from unittest import TestCase

import torch
from hydra.experimental import compose, initialize_config_dir

from code2seq.dataset import PathContextDataset, PathContextBatch
from code2seq.model.modules import PathDecoder
from code2seq.utils.filesystem import get_test_resources_dir
from code2seq.utils.vocabulary import Vocabulary


class TestPathDecoder(TestCase):
    def test_forward(self):
        with initialize_config_dir(config_dir=get_test_resources_dir()):
            config = compose("code2seq-test", overrides=[f"data_folder={get_test_resources_dir()}"])

        dataset_folder = join(config.data_folder, config.dataset.name)
        vocabulary = Vocabulary.load_vocabulary(join(dataset_folder, config.vocabulary_name))
        data_file_path = join(dataset_folder, f"{config.dataset.name}.{config.train_holdout}.c2s")
        dataset = PathContextDataset(data_file_path, config, vocabulary, False)
        batch = PathContextBatch([dataset[i] for i in range(config.hyper_parameters.batch_size)])
        number_of_paths = sum(batch.contexts_per_label)

        model = PathDecoder(config.decoder, len(vocabulary.label_to_id), 0, 0)

        fake_encoder_output = torch.rand(number_of_paths, config.decoder.decoder_size)
        output = model(fake_encoder_output, batch.contexts_per_label, config.dataset.target.max_parts)

        true_shape = (
            config.dataset.target.max_parts,
            config.hyper_parameters.batch_size,
            len(vocabulary.label_to_id),
        )
        self.assertTupleEqual(true_shape, output.shape)
