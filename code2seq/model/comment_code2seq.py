from typing import Dict

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from omegaconf import DictConfig
from torchmetrics import MetricCollection, Metric

from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import CommentVocabulary
from code2seq.model import Code2Seq
from code2seq.model.modules.comment_decoder import CommentDecoder
from code2seq.model.modules.metrics import CommentChrF


class CommentCode2Seq(Code2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: CommentVocabulary,
        teacher_forcing: float = 0.0,
    ):
        super(Code2Seq, self).__init__()

        self.save_hyperparameters()
        self._optim_config = optimizer_config
        self._vocabulary = vocabulary

        tokenizer = vocabulary.tokenizer

        print(tokenizer.convert_ids_to_tokens([225]))
        self._pad_idx = tokenizer.pad_token_id
        self._eos_idx = tokenizer.eos_token_id
        self._sos_idx = tokenizer.bos_token_id
        ignore_idx = [self._sos_idx, tokenizer.unk_token_id]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self._pad_idx, eos_idx=self._eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }

        # TODO add concatenation and rouge-L metric
        metrics.update({f"{holdout}_chrf": CommentChrF(tokenizer) for holdout in ["val", "test"]})
        self._metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        self._decoder = CommentDecoder(
            model_config,
            vocab_size=tokenizer.vocab_size,
            pad_token=self._pad_idx,
            sos_token=self._sos_idx,
            teacher_forcing=teacher_forcing,
        )

        self._loss = SequenceCrossEntropyLoss(self._pad_idx, reduction="seq-mean")

    def _shared_step(self, batch: BatchedLabeledPathContext, step: str) -> Dict:
        target_sequence = batch.labels if step != "test" else None
        # [seq length; batch size; vocab size]
        logits, _ = self.logits_from_batch(batch, target_sequence)
        # if step == "test":
        logits = logits[1:]
        # else:
        #     logits = logits[:-1]
        batch.labels = batch.labels[1:]
        result = {f"{step}/loss": self._loss(logits, batch.labels)}

        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self._metrics[f"{step}_f1"](prediction, batch.labels)
            result.update(
                {f"{step}/f1": metric.f1_score, f"{step}/precision": metric.precision, f"{step}/recall": metric.recall}
            )
            if step != "train":
                result[f"{step}/chrf"] = self._metrics[f"{step}_chrf"](prediction, batch.labels)

        return result
