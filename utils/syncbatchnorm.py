import torch
import torch.nn as nn


def convert_model(model):
    """
    Dummy SyncBatchNorm converter.
    Since we are using single GPU,
    we do not need true synchronized batch norm.

    This function simply returns the model unchanged.
    """
    return model
