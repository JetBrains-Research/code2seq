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

## Dataset mining

To prepare your own dataset with a storage format supported by this implementation, use on the following:
1. Original dataset preprocessing from vanilla repository
2. [`astminer`](https://github.com/JetBrains-Research/astminer):
the tool for mining path-based representation and more with multiple language support.
3. [`PSIMiner`](https://github.com/JetBrains-Research/psiminer):
the tool for extracting PSI trees from IntelliJ Platform and creating datasets from them.
## Available checkpoints

### Method name prediction
| Dataset (with link)                                                                                                     | Checkpoint                                                                                                        | # epochs | F1-score | Precision | Recall | ChrF  |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------|----------|-----------|--------|-------|
| [Java-small](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/java-paths-methods/java-small.tar.gz) | [link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/checkpoints/code2seq_java_small.ckpt) | 11       | 41.49    | 54.26     | 33.59  | 30.21 |
| [Java-med](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/java-paths-methods/java-med.tar.gz)     | [link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/checkpoints/code2seq_java_med.ckpt)   | 10       | 48.17    | 58.87     | 40.76  | 42.32 |

## Configuration

The model is fully configurable by standalone YAML file.
Navigate to [config](config) directory to see examples of configs.

## Examples

Model training may be done via PyTorch Lightning trainer.
See it [documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) for more information.

```python
from argparse import ArgumentParser

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.model import Code2Seq


def train(config: DictConfig):
    # Define data module
    data_module = PathContextDataModule(config.data_folder, config.data)

    # Define model
    model = Code2Seq(
        config.model,
        config.optimizer,
        data_module.vocabulary,
        config.train.teacher_forcing
    )

    # Define hyper parameters
    trainer = Trainer(max_epochs=config.train.n_epochs)

    # Train model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("config", help="Path to YAML configuration file", type=str)
    __args = __arg_parser.parse_args()

    __config = OmegaConf.load(__args.config)
    train(__config)
```
