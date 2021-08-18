from typing import List, Tuple, Iterable

import torch
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def configure_optimizers_alon(
    optim_config: DictConfig, parameters: Iterable[torch.Tensor]
) -> Tuple[List[Optimizer], List[_LRScheduler]]:
    """Create optimizers like in original Alon work
    https://github.com/tech-srl/code2seq/blob/a01076ef649d298e5f90ac2ce1f6a42f4ff49cc2/model.py#L386-L397
    :param optim_config: hyper parameters
    :param parameters: model parameters for optimization
    :return: list of optimizers and schedulers
    """
    optimizer: Optimizer
    if optim_config.optimizer == "Momentum":
        # using the same momentum value as in original realization by Alon
        optimizer = SGD(
            parameters,
            optim_config.lr,
            momentum=0.95,
            nesterov=optim_config.nesterov,
            weight_decay=optim_config.weight_decay,
        )
    elif optim_config.optimizer == "Adam":
        optimizer = Adam(parameters, optim_config.lr, weight_decay=optim_config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_config.optimizer}, try one of: Adam, Momentum")
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: optim_config.decay_gamma ** epoch)
    return [optimizer], [scheduler]
