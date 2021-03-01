# code2seq

[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Github action: build](https://github.com/SpirinEgor/code2seq/workflows/Build/badge.svg)](https://github.com/SpirinEgor/code2seq/actions?query=workflow%3ABuild)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


PyTorch's implementation of code2seq model.

## Installation

You can easily install model through the PIP:
```shell
pip install code2seq
```

## Usage

Minimal code example to run the model:
```python
from os.path import join

import hydra
from code2seq.dataset import PathContextDataModule
from code2seq.model import Code2Seq
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pytorch_lightning import Trainer


@hydra.main(config_path="configs")
def train(config: DictConfig):
    vocabulary_path = join(config.data_folder, config.dataset.name, config.vocabulary_name)
    vocabulary = Vocabulary.load_vocabulary(vocabulary_path)
    model = Code2Seq(config, vocabulary)
    data_module = PathContextDataModule(config, vocabulary)

    trainer = Trainer(max_epochs=config.hyper_parameters.n_epochs)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
```

Navigate to [code2seq/configs](code2seq/configs) to see examples of configs.
If you had any questions then feel free to open the issue.