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
from argparse import ArgumentParser

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.model import Code2Seq


def train(config: DictConfig):
    # Load data module
    data_module = PathContextDataModule(config.data_folder, config.data)
    data_module.prepare_data()
    data_module.setup()

    # Load model
    model = Code2Seq(
        config.model,
        config.optimizer,
        data_module.vocabulary,
        config.train.teacher_forcing
    )

    trainer = Trainer(max_epochs=config.hyper_parameters.n_epochs)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("config", help="Path to YAML configuration file", type=str)
    __args = __arg_parser.parse_args()

    __config = OmegaConf.load(__args.config)
    train(__config)
```

Navigate to [config](config) directory to see examples of configs.
If you have any questions, then feel free to open the issue.