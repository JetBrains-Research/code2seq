from os.path import join
from unittest import TestCase

import torch
from hydra.experimental import compose, initialize_config_dir

from dataset import PathContextDataset, PathContextBatch
from model.modules import PathDecoder
from utils.filesystem import get_test_data_info, get_config_directory
from utils.vocabulary import Vocabulary


class TestPathDecoder(TestCase):
    def test_forward(self):
        with initialize_config_dir(config_dir=get_config_directory()):
            data_folder, dataset_name = get_test_data_info()
            config = compose("main", overrides=[f"data_folder={data_folder}", f"dataset.name={dataset_name}"])

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
