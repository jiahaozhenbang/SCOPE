# encoding: utf-8
import torch
import numpy as np
from typing import List


def collate_to_max_length_with_id(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    tokens_size=[sample[-1] for sample in batch]
    srcs=[sample[-2] for sample in batch]
    ids=[sample[-3] for sample in batch]
    batch=[sample[:-3] for sample in batch]
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full((batch_size, max_lengths[field_idx]),
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields-1)]
    output.append(torch.full((batch_size, max_lengths[-1],3),
                         fill_value=fill_values[-1],
                         dtype=batch[0][-1].dtype))
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    output.append(ids)
    output.append(srcs)
    output.append(tokens_size)
    return output

def collate_to_max_length_for_train_dynamic_pron_loss(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full((batch_size, max_lengths[field_idx]),
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields-1)]
    output.append(torch.full((batch_size, max_lengths[-1],3),
                         fill_value=fill_values[-1],
                         dtype=batch[0][-1].dtype))
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    return output
