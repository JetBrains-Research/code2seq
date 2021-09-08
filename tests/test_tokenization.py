import unittest

import torch

from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.vocabulary import Vocabulary


class TestDatasetTokenization(unittest.TestCase):
    vocab = {Vocabulary.PAD: 0, Vocabulary.UNK: 1, Vocabulary.SOS: 2, Vocabulary.EOS: 3, "my": 4, "super": 5}

    def test_tokenize_label(self):
        raw_label = "my|super|label"
        tokenized = PathContextDataset.tokenize_label(raw_label, self.vocab, 5)
        # <SOS> my super <UNK> <EOS> <PAD>
        correct = [2, 4, 5, 1, 3, 0]

        self.assertListEqual(tokenized, correct)

    def test_tokenize_class(self):
        raw_class = "super"
        tokenized = PathContextDataset.tokenize_class(raw_class, self.vocab)
        correct = [5]

        self.assertListEqual(tokenized, correct)

    def test_tokenize_token(self):
        raw_token = "my|super|token"
        tokenized = PathContextDataset.tokenize_token(raw_token, self.vocab, 5)
        correct = [4, 5, 1, 0, 0]

        self.assertListEqual(tokenized, correct)


if __name__ == "__main__":
    unittest.main()
