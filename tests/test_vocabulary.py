from collections import Counter
from unittest import TestCase

from dataset import Vocabulary


class TestVocabulary(TestCase):
    def test_add_from_counter_all_values(self):
        vocab = Vocabulary()
        new_token_to_id = Counter({"a": 3, "b": 1, "c": 4, "d": 1})
        vocab.add_from_counter("token_to_id", new_token_to_id)
        self.assertDictEqual(vocab.token_to_id, {"c": 0, "a": 1, "b": 2, "d": 3})

    def test_add_from_counter_all_values_with_default(self):
        vocab = Vocabulary()
        new_token_to_id = Counter({"a": 3, "b": 1, "c": 4, "d": 1})
        vocab.add_from_counter("token_to_id", new_token_to_id, add_values=["<SOS>", "<EOS>"])
        self.assertDictEqual(vocab.token_to_id, {"<SOS>": 0, "<EOS>": 1, "c": 2, "a": 3, "b": 4, "d": 5})

    def test_add_from_counter_n_most_values(self):
        vocab = Vocabulary()
        new_token_to_id = Counter({"a": 3, "b": 1, "c": 4, "d": 1})
        vocab.add_from_counter("token_to_id", new_token_to_id, n_most_values=2)
        self.assertDictEqual(vocab.token_to_id, {"c": 0, "a": 1})

    def test_add_from_counter_n_most_values_with_default(self):
        vocab = Vocabulary()
        new_token_to_id = Counter({"a": 3, "b": 1, "c": 4, "d": 1})
        vocab.add_from_counter(
            "token_to_id", new_token_to_id, n_most_values=4, add_values=["<SOS>", "<EOS>"],
        )
        self.assertDictEqual(vocab.token_to_id, {"<SOS>": 0, "<EOS>": 1, "c": 2, "a": 3})

    def test_add_from_counter_raise_error(self):
        vocab = Vocabulary()
        values_counter = Counter({"a": 3, "b": 1, "c": 4, "d": 1})
        with self.assertRaises(ValueError):
            vocab.add_from_counter("unknown_field", values_counter)
