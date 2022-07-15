from typing import Dict

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics import MetricCollection, Metric
from transformers import RobertaTokenizerFast

from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import Vocabulary
from code2seq.model import Code2Seq


class CommentCode2Seq(Code2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: Vocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)

        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
        self.__pad_idx = tokenizer.pad_token_id
        eos_idx = tokenizer.eos_token_id
        ignore_idx = [tokenizer.bos_token_id, tokenizer.unk_token_id]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self.__pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }
        # TODO add chrf back
        self.__metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        output_size = len(tokenizer.get_vocab())
        decoder_step = LSTMDecoderStep(model_config, output_size, self.__pad_idx)
        self._decoder = Decoder(decoder_step, output_size, tokenizer.eos_token_id, teacher_forcing)

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")

    def _shared_step(self, batch: BatchedLabeledPathContext, step: str) -> Dict:
        target_sequence = batch.labels if step == "train" else None
        # [seq length; batch size; vocab size]
        logits, _ = self.logits_from_batch(batch, target_sequence)
        result = {f"{step}/loss": self.__loss(logits[1:], batch.labels[1:])}

        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self.__metrics[f"{step}_f1"](prediction, batch.labels)
            result.update(
                {f"{step}/f1": metric.f1_score, f"{step}/precision": metric.precision, f"{step}/recall": metric.recall}
            )

        return result

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        with torch.no_grad():
            losses = [so if isinstance(so, torch.Tensor) else so["loss"] for so in step_outputs]
            mean_loss = torch.stack(losses).mean()
            metric = self.__metrics[f"{step}_f1"].compute()
            log = {
                f"{step}/loss": mean_loss,
                f"{step}/f1": metric.f1_score,
                f"{step}/precision": metric.precision,
                f"{step}/recall": metric.recall,
            }
            self.__metrics[f"{step}_f1"].reset()
        self.log_dict(log, on_step=False, on_epoch=True)
