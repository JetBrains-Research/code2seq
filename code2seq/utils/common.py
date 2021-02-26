from typing import Dict, List
from warnings import filterwarnings

from omegaconf import DictConfig


def print_table(data: Dict[str, List[str]]):
    row_lens = [max(len(header), max([len(s) for s in values])) for header, values in data.items()]
    row_template = " | ".join(["{:<" + str(i) + "}" for i in row_lens])
    headers = [key for key in data.keys()]
    max_data_per_col = max([len(v) for v in data.values()])
    row_data = []
    for i in range(max_data_per_col):
        row_data.append([v[i] if len(v) > i else "" for k, v in data.items()])

    header_line = row_template.format(*headers)
    delimiter_line = "-" * len(header_line)
    row_lines = [row_template.format(*row) for row in row_data]
    print("", header_line, delimiter_line, *row_lines, sep="\n")


def print_config(config: DictConfig):
    known_config_fields = ["hyper_parameters", "encoder", "decoder", "classifier"]
    config_data = {}
    for column in known_config_fields:
        if column not in config:
            continue
        config_data[column] = [f"{k}: {v}" for k, v in config[column].items()]
    config_data["dataset"] = [f"name: {config.dataset.name}"]
    for key, val in config.dataset.items():
        if isinstance(val, DictConfig):
            config_data["dataset"] += [f"{key}.{k}: {v}" for k, v in val.items()]

    print_table(config_data)


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities.distributed", lineno=50)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load
