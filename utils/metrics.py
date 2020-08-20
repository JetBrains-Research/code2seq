from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class SubtokenStatistic:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, other_statistic: "SubtokenStatistic"):
        self.true_positive += other_statistic.true_positive
        self.false_positive += other_statistic.false_positive
        self.false_negative += other_statistic.false_negative

    def calculate_metrics(self, group: str = None) -> Dict[str, int]:
        precision, recall, f1 = 0, 0, 0
        if self.true_positive + self.false_positive > 0:
            precision = self.true_positive / (self.true_positive + self.false_positive)
        if self.true_positive + self.false_negative > 0:
            recall = self.true_positive / (self.true_positive + self.false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
        if group is not None:
            for key in list(metrics_dict.keys()):
                metrics_dict[f"{group}/{key}"] = metrics_dict.pop(key)
        return metrics_dict

    @staticmethod
    def calculate_statistic(
        true_subtokens: torch.Tensor, predicted_subtokens: torch.Tensor, skip: List[int] = None
    ) -> "SubtokenStatistic":
        """Calculate subtoken statistic for ground truth and predicted batches of labels.

        :param true_subtokens: [true seq length; batch size] ground truth labels
        :param predicted_subtokens: [pred seq length; batch size] predicted labels
        :param skip: list of subtokens ids that should be ignored
        :return: dataclass with calculated statistic
        """
        if true_subtokens.shape[1] != predicted_subtokens.shape[1]:
            raise ValueError(
                f"unequal batch sizes for ground truth ({true_subtokens.shape[1]}) subtokens"
                f"and predicted ({predicted_subtokens.shape[1]})"
            )
        if skip is None:
            skip = []
        subtoken_statistic = SubtokenStatistic()
        batch_size = true_subtokens.shape[1]
        for batch_idx in range(batch_size):
            gt_seq = [st for st in true_subtokens[:, batch_idx] if st not in skip]
            pred_seq = [st for st in predicted_subtokens[:, batch_idx] if st not in skip]

            if len(gt_seq) == len(pred_seq) and all([g == p for g, p in zip(gt_seq, pred_seq)]):
                subtoken_statistic.true_positive += len(gt_seq)
                continue

            for pred_subtoken in pred_seq:
                if pred_subtoken in gt_seq:
                    subtoken_statistic.true_positive += 1
                else:
                    subtoken_statistic.false_positive += 1
            for gt_subtoken in gt_seq:
                if gt_subtoken not in pred_seq:
                    subtoken_statistic.false_negative += 1
        return subtoken_statistic

    @staticmethod
    def union_statistics(stats: List["SubtokenStatistic"]) -> "SubtokenStatistic":
        union_subtoken_statistic = SubtokenStatistic()
        for stat in stats:
            union_subtoken_statistic.update(stat)
        return union_subtoken_statistic
