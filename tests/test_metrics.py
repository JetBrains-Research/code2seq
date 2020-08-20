from unittest import TestCase

import torch

from utils.metrics import SubtokenStatistic


class TestSubtokenStatistic(TestCase):
    def test_update(self):
        st_stat = SubtokenStatistic(1, 2, 3)
        st_stat_other = SubtokenStatistic(4, 5, 6)
        st_stat.update(st_stat_other)

        self.assertEqual(st_stat.true_positive, 5)
        self.assertEqual(st_stat.false_positive, 7)
        self.assertEqual(st_stat.false_negative, 9)

    def test_calculate_metrics(self):
        st_stat = SubtokenStatistic(3, 7, 2)
        metrics = st_stat.calculate_metrics()
        true_metrics = {"precision": 0.3, "recall": 0.6, "f1": 0.4}

        self.assertDictEqual(metrics, true_metrics)

    def test_calculate_metrics_with_group(self):
        st_stat = SubtokenStatistic(3, 7, 2)
        metrics = st_stat.calculate_metrics(group="train")
        true_metrics = {"train/precision": 0.3, "train/recall": 0.6, "train/f1": 0.4}

        self.assertDictEqual(metrics, true_metrics)

    def test_calculate_zero_metrics(self):
        st_stat = SubtokenStatistic(0, 0, 0)
        metrics = st_stat.calculate_metrics()
        true_metrics = {"precision": 0, "recall": 0, "f1": 0}

        self.assertDictEqual(metrics, true_metrics)

    def test_calculate_statistic(self):
        gt_subtokens = torch.tensor([[1, 1, 1, 0], [2, 2, 0, -1], [3, 3, -1, -1], [-1, -1, -1, -1]])
        pred_subtokens = torch.tensor([[2, 4, 1, 0], [4, 5, 2, 0], [1, 6, 3, 0], [5, -1, -1, -1]])
        skip = [-1, 0]

        st_stat = SubtokenStatistic.calculate_statistic(gt_subtokens, pred_subtokens, skip)

        self.assertEqual(st_stat.true_positive, 3)
        self.assertEqual(st_stat.false_positive, 7)
        self.assertEqual(st_stat.false_negative, 4)

    def test_calculate_statistic_equal_tensors(self):
        gt_subtokens = torch.tensor([[1, 2, 3, 4, 5, 0, -1]])
        pred_subtokens = torch.tensor([[1, 2, 3, 4, 5, 0, -1]])
        skip = [-1, 0]

        st_stat = SubtokenStatistic.calculate_statistic(gt_subtokens, pred_subtokens, skip)

        self.assertEqual(st_stat.true_positive, 5)
        self.assertEqual(st_stat.false_positive, 0)
        self.assertEqual(st_stat.false_negative, 0)
