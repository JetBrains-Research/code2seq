from unittest import TestCase

import torch

from code2seq.utils.metrics import PredictionStatistic


class TestPredictionStatistic(TestCase):
    @staticmethod
    def _create_statistic_with_values(tp: int, fp: int, fn: int) -> PredictionStatistic:
        statistic = PredictionStatistic(False)
        statistic._true_positive = tp
        statistic._false_positive = fp
        statistic._false_negative = fn
        return statistic

    def test_update(self):
        stat1 = self._create_statistic_with_values(1, 2, 3)
        stat2 = self._create_statistic_with_values(4, 5, 6)
        union = PredictionStatistic.create_from_list([stat1, stat2])

        self.assertEqual(union._true_positive, 5)
        self.assertEqual(union._false_positive, 7)
        self.assertEqual(union._false_negative, 9)

    def test_calculate_metrics(self):
        stat = self._create_statistic_with_values(3, 7, 2)
        metrics = stat.get_metric()
        true_metrics = {"precision": 0.3, "recall": 0.6, "f1": 0.4}

        self.assertDictEqual(metrics, true_metrics)

    def test_calculate_zero_metrics(self):
        stat = self._create_statistic_with_values(0, 0, 0)
        metrics = stat.get_metric()
        true_metrics = {"precision": 0, "recall": 0, "f1": 0}

        self.assertDictEqual(metrics, true_metrics)

    def test_calculate_statistic(self):
        gt_subtokens = torch.tensor([[1, 1, 1, 0], [2, 2, 0, -1], [3, 3, -1, -1], [-1, -1, -1, -1]])
        pred_subtokens = torch.tensor([[2, 4, 1, 0], [4, 5, 2, 0], [1, 6, 3, 0], [5, -1, -1, -1]])
        skip = [-1, 0]

        statistic = PredictionStatistic(False, skip_tokens=skip)
        statistic.update_statistic(gt_subtokens, pred_subtokens)

        self.assertEqual(statistic._true_positive, 3)
        self.assertEqual(statistic._false_positive, 7)
        self.assertEqual(statistic._false_negative, 4)

    def test_calculate_statistic_equal_tensors(self):
        gt_subtokens = torch.tensor([1, 2, 3, 4, 5, 0, -1]).view(-1, 1)
        pred_subtokens = torch.tensor([1, 2, 3, 4, 5, 0, -1]).view(-1, 1)
        skip = [-1, 0]

        statistic = PredictionStatistic(False, skip_tokens=skip)
        statistic.update_statistic(gt_subtokens, pred_subtokens)

        self.assertEqual(statistic._true_positive, 5)
        self.assertEqual(statistic._false_positive, 0)
        self.assertEqual(statistic._false_negative, 0)

    def test_calculate_statistic_with_masking(self):
        gt_subtokens = torch.tensor([1, 2, 3, 6, 7, 8, 0, 0, 0]).view(-1, 1)
        pred_subtokens = torch.tensor([1, 2, 3, 4, 5, 0, 6, 0, 8]).view(-1, 1)

        statistic = PredictionStatistic(True, 0, [0])
        statistic.update_statistic(gt_subtokens, pred_subtokens)

        self.assertEqual(statistic._true_positive, 3)
        self.assertEqual(statistic._false_positive, 2)
        self.assertEqual(statistic._false_negative, 3)

    def test_calculate_statistic_with_masking_long_sequence(self):
        gt_subtokens = torch.tensor([1, 2, 3, 6, 7, 8, 0, 0, 0]).view(-1, 1)
        pred_subtokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(-1, 1)

        statistic = PredictionStatistic(True, 0, [0])
        statistic.update_statistic(gt_subtokens, pred_subtokens)

        self.assertEqual(statistic._true_positive, 6)
        self.assertEqual(statistic._false_positive, 2)
        self.assertEqual(statistic._false_negative, 0)
