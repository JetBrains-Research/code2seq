from typing import Dict

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from sacrebleu import CHRF
from torchmetrics import MetricCollection, Metric
from transformers import RobertaTokenizerFast

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

        # TODO add concatenation and rouge-L metric
        metrics.update(
            {f"{holdout}_chrf": CommentChrF(tokenizer) for holdout in ["val", "test"]}
        )
        self._metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        output_size = len(tokenizer.get_vocab())
        decoder_step = LSTMDecoderStep(model_config, output_size, self.__pad_idx)
        self._decoder = Decoder(decoder_step, output_size, tokenizer.eos_token_id, teacher_forcing)

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")


class CommentChrF(Metric):
    def __init__(self, tokenizer: RobertaTokenizerFast, **kwargs):
        super().__init__(**kwargs)
        self.__tokenizer = tokenizer
        self.__chrf = CHRF()

        # Metric states
        self.add_state("chrf", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculated ChrF metric on predicted tensor w.r.t. target tensor.

        :param predicted: [pred seq len; batch size] -- tensor with predicted tokens
        :param target: [target seq len; batch size] -- tensor with ground truth tokens
        :return:
        """
        batch_size = target.shape[1]
        if predicted.shape[1] != batch_size:
            raise ValueError(f"Wrong batch size for prediction (expected: {batch_size}, actual: {predicted.shape[1]})")

        for batch_idx in range(batch_size):
            target_seq = [token.item() for token in target[:, batch_idx]]
            predicted_seq = [token.item() for token in predicted[:, batch_idx]]

            target_str = " ".join(self.__tokenizer.convert_ids_to_tokens(target_seq, skip_special_tokens=True))
            predicted_str = " ".join(self.__tokenizer.convert_ids_to_tokens(predicted_seq, skip_special_tokens=True))
            print(target_str, "||", predicted_str)

            if target_str == "":
                # Empty target string mean that the original string encoded only with <UNK> token
                continue

            self.chrf += self.__chrf.sentence_score(predicted_str, [target_str]).score
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.chrf / self.count
