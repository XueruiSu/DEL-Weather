"""Iterative datapipes to read weather dataset in seqrecord format"""

import numpy as np
import torch
import torchdata.datapipes as datapipe
from torchdata.datapipes.iter import IterableWrapper
from typing import Callable, Dict, List, Optional
from data.seqrecord_utils.wseqrecord import WSeqRecord
from torchvision.transforms import transforms
from functools import partial
from data.datapipes import (
    # Import functional form of datapipe operations.
    # The linter is confused about these, so we disable unused import warnings.
    FileMetadataFromWSeqRecord,  # noqa: F401
    FramePairsFromWSeqRecord,  # noqa: F401
    SeparateSingleAtmosVars,  # noqa: F401
)


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    single_transforms: Optional[transforms.Normalize] = None,
    atmos_transforms: Optional[transforms.Normalize] = None,
):
    if batch[0]["single_input"] is None:
        single_inputs = None
    else:
        single_inputs = torch.stack([batch[i]["single_input"] for i in range(len(batch))], axis=0)
        if single_transforms is not None:
            single_inputs = single_transforms(single_inputs.squeeze(1))
            # (batch, 1, var_cnt, H, W)
            single_inputs = single_inputs.unsqueeze(1)

    atmos_keys = batch[0]["atmos_input"].keys()
    atmos_inputs = {
        k: torch.stack([batch[i]["atmos_input"][k] for i in range(len(batch))], axis=0)
        for k in atmos_keys
    }
    if atmos_transforms is not None:
        atmos_inputs = {
            k: atmos_transforms[k](v.squeeze(1)).unsqueeze(1) for k, v in atmos_inputs.items()
        }
    if batch[0]["single_target"] is None:
        single_targets = None
    else:
        single_targets = torch.stack([batch[i]["single_target"] for i in range(len(batch))], axis=0)
        if single_transforms is not None:
            # (batch, var_cnt, H, W)
            single_targets = single_transforms(single_targets)

    atmos_targets = {
        k: torch.stack([batch[i]["atmos_target"][k] for i in range(len(batch))], axis=0)
        for k in atmos_keys
    }
    if atmos_transforms is not None:
        atmos_targets = {k: atmos_transforms[k](v.squeeze(1)) for k, v in atmos_targets.items()}

    lead_times = torch.stack([batch[i]["lead_times"] for i in range(len(batch))], axis=0)
    metadata = batch[0]["metadata"]
    assert metadata == batch[-1]["metadata"]
    return single_inputs, atmos_inputs, single_targets, atmos_targets, lead_times, metadata


def build_wdatapipe(
    record: WSeqRecord,
    file_shuffle_buffer_size: Optional[int],
    single_vars: List[str],
    atmos_vars: List[str],
    lat: np.ndarray,
    lon: np.ndarray,
    hrs_each_step: int,
    batch_size: int,
    single_transforms: Optional[transforms.Normalize] = None,
    atmos_transforms: Optional[transforms.Normalize] = None,
    mappings: List[Callable] = [],
    prefetch: int = 0,
) -> datapipe.iter.IterDataPipe:
    """Iteratively apply operations to datapipe: shuffle, sharding, map, batch, collator

    Args:
        datapipe (datapipe.datapipe.IterDataPipe): entry datapipe
        shuffle_buffer_size (Optional[int]): buffer size for pseudo-shuffle
        batch_size (int):
        mappings (List[Callable]): a list of transforms applied to datapipe, between sharding and batch

    Returns:
        datapipe.datapipe.IterDataPipe: transformed datapipe ready to be sent to dataloader
    """
    # Avoid sending the record object through the pipe as it is very large and slows down the pipe.
    dp = IterableWrapper(list(range(record.num_files))).gen_file_metadata(record=record)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    if file_shuffle_buffer_size is not None:
        dp = dp.shuffle(buffer_size=file_shuffle_buffer_size)
    # Sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating
    # these expensive operations across worker/distributed processes.
    dp = dp.sharding_filter()
    dp = dp.gen_framepair(record=record)
    if prefetch > 0:
        dp = dp.prefetch(prefetch)
    dp = dp.separate_single_atmos_vars(
        single_vars, atmos_vars, lat, lon, hrs_each_step=hrs_each_step
    )
    for i, mapping in enumerate(mappings):
        dp = dp.map(fn=mapping)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    dp = dp.batch(batch_size=batch_size, drop_last=True)
    partial_collate_fn = partial(
        collate_fn, single_transforms=single_transforms, atmos_transforms=atmos_transforms
    )
    dp = dp.collate(collate_fn=partial_collate_fn)
    return dp



def build_wdatapipe_UQ(
    Main_Model_1: torch.nn.modules, 
    Main_Model_2: torch.nn.modules, 
    record: WSeqRecord,
    file_shuffle_buffer_size: Optional[int],
    single_vars: List[str],
    atmos_vars: List[str],
    lat: np.ndarray,
    lon: np.ndarray,
    hrs_each_step: int,
    batch_size: int,
    single_transforms: Optional[transforms.Normalize] = None,
    atmos_transforms: Optional[transforms.Normalize] = None,
    mappings: List[Callable] = [],
    prefetch: int = 0,
) -> datapipe.iter.IterDataPipe:
    """Iteratively apply operations to datapipe: shuffle, sharding, map, batch, collator

    Args:
        datapipe (datapipe.datapipe.IterDataPipe): entry datapipe
        shuffle_buffer_size (Optional[int]): buffer size for pseudo-shuffle
        batch_size (int):
        mappings (List[Callable]): a list of transforms applied to datapipe, between sharding and batch

    Returns:
        datapipe.datapipe.IterDataPipe: transformed datapipe ready to be sent to dataloader
    """
    # Avoid sending the record object through the pipe as it is very large and slows down the pipe.
    dp = IterableWrapper(list(range(record.num_files))).gen_file_metadata(record=record)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    if file_shuffle_buffer_size is not None:
        dp = dp.shuffle(buffer_size=file_shuffle_buffer_size)
    # Sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating
    # these expensive operations across worker/distributed processes.
    dp = dp.sharding_filter()
    dp = dp.gen_framepair(record=record)
    if prefetch > 0:
        dp = dp.prefetch(prefetch)
    dp = dp.separate_single_atmos_vars_UQ(
        single_vars, atmos_vars, lat, lon, hrs_each_step=hrs_each_step, Main_Model_1=Main_Model_1, Main_Model_2=Main_Model_2,
    )
    for i, mapping in enumerate(mappings):
        dp = dp.map(fn=mapping)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    dp = dp.batch(batch_size=batch_size, drop_last=True)
    partial_collate_fn = partial(
        collate_fn, single_transforms=single_transforms, atmos_transforms=atmos_transforms
    )
    dp = dp.collate(collate_fn=partial_collate_fn)
    return dp
