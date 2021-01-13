# code2seq

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyTorch's implementation of code2seq model.

## Configuration

Use `yaml` files from [config](./configs) directory to configure all processes.
`model` option is used to define model, for now repository supports:
- code2seq
- typed-code2seq
- code2class

`data_folder` stands for the path to the folder with dataset.
For checkpoints with predefined config, users can specify data folder by argument in corresponding script.

## Data

Code2seq implementation supports the same data format as the original [model](https://github.com/tech-srl/code2seq).
The only one different is storing vocabulary. To recollect vocabulary use
```shell
PYTHONPATH='.' python preprocessing/build_vocabulary.py
```

## Train model

To train model use `train.py` script
```shell
python train.py model
```
Use [`main.yaml`](configs/main.yaml) to set up hyper-parameters.
Use corresponding configuration from [`configs/model`](configs/model) to set up dataset.

To resume training from saved checkpoint use `--resume` argument
```shell
python train.py model --resume checkpoint.ckpt
```

## Evaluate model

To evaluate trained model use `test.py` script
```shell
python test.py checkpoint.py
```

To specify the folder with data (in case on evaluating on different from training machine) use `--data-folder` argument
```shell
python test.py checkpoint.py --data-folder path
```

