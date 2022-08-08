from typing import Dict

from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from torchmetrics import MetricCollection, Metric

from code2seq.data.vocabulary import CommentVocabulary
from code2seq.model import Code2Seq
from code2seq.model.modules.metrics import CommentChrF


class CommentCode2Seq(Code2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: CommentVocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)

        tokenizer = vocabulary.tokenizer
        self._pad_idx = tokenizer.pad_token_id
        eos_idx = tokenizer.eos_token_id
        ignore_idx = [tokenizer.bos_token_id, tokenizer.unk_token_id]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self._pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }

        # TODO add concatenation and rouge-L metric
        metrics.update({f"{holdout}_chrf": CommentChrF(tokenizer) for holdout in ["val", "test"]})
        self._metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        output_size = tokenizer.vocab_size
        decoder_step = LSTMDecoderStep(model_config, output_size, self._pad_idx)
        self._decoder = Decoder(decoder_step, output_size, tokenizer.eos_token_id, teacher_forcing)

        self._loss = SequenceCrossEntropyLoss(self._pad_idx, reduction="batch-mean")
