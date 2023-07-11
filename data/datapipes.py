"""Iterative datapipes for weather datasets."""

import torchdata.datapipes as datapipe
import numpy as np
import torch

from typing import Dict, List, Tuple
from data.seqrecord_utils.wseqrecord import WSeqRecord
from data import Metadata
from data.seqrecord_utils.wseqrecord import SeqRecordFileMetadata


@datapipe.functional_datapipe("gen_framepair")
class FramePairsFromWSeqRecord(datapipe.iter.IterDataPipe):
    """A torch datapipe that generates input-target frame pairs from seqrecord files."""

    def __init__(
        self,
        source_dp: datapipe.iter.IterDataPipe,
        record: WSeqRecord,
    ) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.record = record

    def __iter__(self):
        yield from WSeqRecord.fast_iterate_framepairs_from_files(self.record, self.source_dp)


@datapipe.functional_datapipe("gen_file_metadata")
class FileMetadataFromWSeqRecord(datapipe.iter.IterDataPipe):
    """A torch datapipe that generates file metadata from seqrecord file ids."""

    def __init__(self, dp: datapipe.iter.IterDataPipe, record) -> None:
        super().__init__()
        self.dp = dp
        self.record = record

    def __iter__(self):
        for fileidx in self.dp:
            yield SeqRecordFileMetadata(self.record, fileidx)


@datapipe.functional_datapipe("separate_single_atmos_vars")
class SeparateSingleAtmosVars(datapipe.iter.IterDataPipe):
    """A Torch datapipe that reads frames and concatenates the contents of the frames into big
    tensors. It produces an output comprising single-level variables and multi-output atmospheric
    variables for the input and the target together with additional metadata."""

    def __init__(
        self,
        data_dp: datapipe.iter.IterDataPipe,
        single_vars: Tuple[str],
        atmos_vars: Tuple[str],
        latitude: np.ndarray,
        longitude: np.ndarray,
        hrs_each_step: int,
    ) -> None:
        super().__init__()
        self.data_dp = data_dp
        # TODO(Cris): Test that the ordering of the single_vars and atmos_vars is correct.
        self.single_vars = single_vars
        atmos_variables = {"_".join(k.split("_")[:-1]) for k in atmos_vars}
        atmos_levels: Dict[str, List[int]] = {k: [] for k in atmos_variables}
        for k in atmos_vars:
            atmos_levels["_".join(k.split("_")[:-1])].append(int(k.split("_")[-1]))

        self.all_atmos_vars = atmos_vars
        self.atmos_variables = atmos_variables
        self.atmos_levels = atmos_levels
        self.hrs_each_step = hrs_each_step
        self.latitude = torch.from_numpy(latitude)
        self.longitude = torch.from_numpy(longitude)

    def __iter__(self):
        for data in self.data_dp:
            input_feats = data["input"]
            target_feats = data["target"]
            input_features = data["input_features"]
            target_features = data["target_features"]
            # TODO: handle this better later
            if input_feats.shape[-2:] == (721, 1440):
                input_feats = input_feats[..., :-1, :]
                target_feats = target_feats[..., :-1, :]

            if len(input_feats.shape) == 3:
                # Adding T dimension
                input_feats = input_feats[None, ...]

            assert input_features == target_features
            # find single_vars indexs in input_features
            single_vars_idx = []
            for var in self.single_vars:
                single_vars_idx.append(input_features.index(var))
            # find atmos_vars indexs in input_features
            atmos_vars2idx = {}
            for var in self.atmos_variables:
                for level in self.atmos_levels[var]:
                    atmos_vars2idx[f"{var}_{level}"] = input_features.index(f"{var}_{level}")

            single_input = input_feats[:, single_vars_idx]

            atmos_input = {}
            for var in self.atmos_variables:
                atmos_input[var] = []
                for level in self.atmos_levels[var]:
                    atmos_input[f"{var}"].append(input_feats[:, atmos_vars2idx[f"{var}_{level}"]])

                atmos_input[var] = torch.stack(atmos_input[var], axis=1)

            single_target = target_feats[single_vars_idx]
            atmos_target = {}
            for var in self.atmos_variables:
                atmos_target[var] = []
                for level in self.atmos_levels[var]:
                    atmos_target[f"{var}"].append(target_feats[atmos_vars2idx[f"{var}_{level}"]])

                atmos_target[var] = torch.stack(atmos_target[var], axis=0)

            single_keys = tuple(self.single_vars)
            atmos_var_levels = self.atmos_levels

            lead_times = data["lookahead_steps"].to(dtype=input_feats.dtype) * self.hrs_each_step

            yield {
                "single_input": single_input,
                "atmos_input": atmos_input,
                "single_target": single_target,
                "atmos_target": atmos_target,
                "lead_times": lead_times,
                "metadata": Metadata(single_keys, atmos_var_levels, self.latitude, self.longitude),
            }


@datapipe.functional_datapipe("separate_single_atmos_vars_UQ")
class SeparateSingleAtmosVars_UQ(datapipe.iter.IterDataPipe):
    """A Torch datapipe that reads frames and concatenates the contents of the frames into big
    tensors. It produces an output comprising single-level variables and multi-output atmospheric
    variables for the input and the target together with additional metadata."""

    def __init__(
        self,
        data_dp: datapipe.iter.IterDataPipe,
        single_vars: Tuple[str],
        atmos_vars: Tuple[str],
        latitude: np.ndarray,
        longitude: np.ndarray,
        hrs_each_step: int,
        
        Main_Model_1: torch.nn.modules, 
        Main_Model_2: torch.nn.modules, 
    ) -> None:
        super().__init__()
        self.data_dp = data_dp
        # TODO(Cris): Test that the ordering of the single_vars and atmos_vars is correct.
        self.single_vars = single_vars
        atmos_variables = {"_".join(k.split("_")[:-1]) for k in atmos_vars}
        atmos_levels: Dict[str, List[int]] = {k: [] for k in atmos_variables}
        for k in atmos_vars:
            atmos_levels["_".join(k.split("_")[:-1])].append(int(k.split("_")[-1]))

        self.all_atmos_vars = atmos_vars
        self.atmos_variables = atmos_variables
        self.atmos_levels = atmos_levels
        self.hrs_each_step = hrs_each_step
        self.latitude = torch.from_numpy(latitude)
        self.longitude = torch.from_numpy(longitude)
        
        self.Main_Model_1 = Main_Model_1
        self.Main_Model_2 = Main_Model_2

    def __iter__(self):
        for data in self.data_dp:
            input_feats = data["input"]
            target_feats = data["target"]
            input_features = data["input_features"]
            target_features = data["target_features"]
            # TODO: handle this better later
            if input_feats.shape[-2:] == (721, 1440):
                input_feats = input_feats[..., :-1, :]
                target_feats = target_feats[..., :-1, :]

            if len(input_feats.shape) == 3:
                # Adding T dimension
                input_feats = input_feats[None, ...]

            assert input_features == target_features
            # find single_vars indexs in input_features
            single_vars_idx = []
            for var in self.single_vars:
                single_vars_idx.append(input_features.index(var))
            # find atmos_vars indexs in input_features
            atmos_vars2idx = {}
            for var in self.atmos_variables:
                for level in self.atmos_levels[var]:
                    atmos_vars2idx[f"{var}_{level}"] = input_features.index(f"{var}_{level}")

            single_input = input_feats[:, single_vars_idx]

            atmos_input = {}
            for var in self.atmos_variables:
                atmos_input[var] = []
                for level in self.atmos_levels[var]:
                    atmos_input[f"{var}"].append(input_feats[:, atmos_vars2idx[f"{var}_{level}"]])

                atmos_input[var] = torch.stack(atmos_input[var], axis=1)

            single_target = target_feats[single_vars_idx]
            atmos_target = {}
            for var in self.atmos_variables:
                atmos_target[var] = []
                for level in self.atmos_levels[var]:
                    atmos_target[f"{var}"].append(target_feats[atmos_vars2idx[f"{var}_{level}"]])

                atmos_target[var] = torch.stack(atmos_target[var], axis=0)

            single_keys = tuple(self.single_vars)
            atmos_var_levels = self.atmos_levels

            lead_times = data["lookahead_steps"].to(dtype=input_feats.dtype) * self.hrs_each_step

            # UQ_inference:
            metadata = Metadata(single_keys, atmos_var_levels, self.latitude, self.longitude)
            
            _, single_pre_1, atmos_pre_1 = self.Main_Model_1(single_input, atmos_input, lead_times, metadata)
            _, single_pre_2, atmos_pre_2 = self.Main_Model_2(single_input, atmos_input, lead_times, metadata)
        
            
            yield {
                "single_input": single_input,
                "atmos_input": atmos_input,
                "single_target": single_target,
                "atmos_target": atmos_target,
                "lead_times": lead_times,
                "metadata": metadata,
                "single_pre_1": single_pre_1, 
                "atmos_pre_1": atmos_pre_1, 
                "single_pre_2": single_pre_2, 
                "atmos_pre_2": atmos_pre_2, 
            }


