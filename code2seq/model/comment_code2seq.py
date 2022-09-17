from typing import Dict

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from torchmetrics import MetricCollection, Metric

from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import CommentVocabulary
from code2seq.model import Code2Seq
from code2seq.model.modules.transformer_comment_decoder import TransformerCommentDecoder
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
        self._decoder = self.get_decoder(model_config, tokenizer.vocab_size, teacher_forcing)

        self._loss = SequenceCrossEntropyLoss(self._pad_idx, reduction="seq-mean")

    def get_decoder(self, model_config: DictConfig, vocab_size: int, teacher_forcing: float) -> torch.nn.Module:
        if model_config.decoder_type == "LSTM":
            decoder_step = LSTMDecoderStep(model_config, vocab_size, self._pad_idx)
            return Decoder(decoder_step, vocab_size, self._sos_idx, teacher_forcing)
        elif model_config.decoder_type == "Transformer":
            return TransformerCommentDecoder(
                model_config,
                vocab_size=vocab_size,
                pad_token=self._pad_idx,
                sos_token=self._sos_idx,
                eos_token=self._eos_idx,
                teacher_forcing=teacher_forcing,
            )
        else:
            raise ValueError

    def _shared_step(self, batch: BatchedLabeledPathContext, step: str) -> Dict:
        target_sequence = batch.labels if step != "test" else None
        # [seq length; batch size; vocab size]
        logits, _ = self.logits_from_batch(batch, target_sequence)
        logits = logits[:-1]
        batch.labels = batch.labels[1:]
        result = {f"{step}/loss": self._loss(logits, batch.labels)}

        with torch.no_grad():
            if step != "train":
                prediction = logits.argmax(-1)
                metric: ClassificationMetrics = self._metrics[f"{step}_f1"](prediction, batch.labels)
                result.update(
                    {
                        f"{step}/f1": metric.f1_score,
                        f"{step}/precision": metric.precision,
                        f"{step}/recall": metric.recall,
                    }
                )
                result[f"{step}/chrf"] = self._metrics[f"{step}_chrf"](prediction, batch.labels)
            else:
                result.update({f"{step}/f1": 0, f"{step}/precision": 0, f"{step}/recall": 0})

        return result
