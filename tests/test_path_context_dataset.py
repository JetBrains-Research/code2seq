from os.path import join
from unittest import TestCase

from dataset.path_context_dataset import PathContextDataset
from tests.tools import get_path_to_test_data
from utils.common import FROM_TOKEN, PATH_TYPES, TO_TOKEN


class TestPathContextDataset(TestCase):

    _data_path = join(get_path_to_test_data(), "train")
    _max_context = 5
    _random_context = True
    _shuffle = True

    def test_taking_next_item(self):
        dataset = PathContextDataset(self._data_path, self._max_context, self._random_context, self._shuffle)
        total_samples = 0

        for x, y, n_paths in dataset:
            self.assertCountEqual([FROM_TOKEN, PATH_TYPES, TO_TOKEN], x.keys())
            for key, value in x.items():
                self.assertLessEqual(value.shape[1], self._max_context)
            total_samples += 1

        self.assertEqual(dataset.get_n_samples(), total_samples)
