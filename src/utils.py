# -*- coding:UTF-8 -*-
import torch
import logging
import deepspeed
import torch.distributed as dist
from contextlib import contextmanager
import json

def get_ds_config(config_path):
    with open(config_path, "r", encoding="utf8") as fr:
        data = json.load(fr)
    return data

class LogMessage:
    def __init__(self, is_distributed):
        self.is_distributed = is_distributed

    def __call__(self, message):
        if (self.is_distributed and dist.get_rank() == 0) or not self.is_distributed:
            logging.info(message)


@contextmanager
def torch_distributed_master_process_first(rank: int):
    """
    Decorator to make all processes in distributed training wait for each master to do something.
    """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if rank == 0:
        torch.distributed.barrier()
