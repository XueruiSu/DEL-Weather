import os
from typing import Optional, Dict, List
import numpy as np
import torch
import pickle
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms

from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    SequentialReadingService,
)


from torchdata.dataloader2 import MultiProcessingReadingService as MPRS
from data.seqrecord_utils.wseqrecord import WSeqRecord
from data.seqrecord.era5_dp import build_wdatapipe


def get_normalization_transforms(
    normalize_path, single_vars: tuple[str, ...], atmos_vars: tuple[str, ...]
):
    mean_path = os.path.join(normalize_path, "normalize_mean.pkl")
    std_path = os.path.join(normalize_path, "normalize_std.pkl")
    if not os.path.exists(mean_path):
        return None, None

    with open(mean_path, "rb") as fp:
        normalize_mean = pickle.load(fp)
    with open(std_path, "rb") as fp:
        normalize_std = pickle.load(fp)

    single_mean = []
    single_std = []
    for var in single_vars:
        if var != "tp":
            single_mean.append(normalize_mean[var])
        else:
            single_mean.append(np.array([0.0]))

        single_std.append(normalize_std[var])

    normalize_single_mean = np.array(single_mean)
    normalize_single_std = np.array(single_std)
    single_transforms = transforms.Normalize(normalize_single_mean, normalize_single_std)

    atmos_variables = {v.rsplit("_", 1)[0] for v in atmos_vars}
    atmos_levels: Dict[str, List[int]] = {k: [] for k in atmos_variables}
    for var in atmos_vars:
        atmos_levels[var.rsplit("_", 1)[0]].append(int(var.rsplit("_", 1)[1]))

    atmos_mean: Dict[str, List] = {k: [] for k in atmos_variables}
    atmos_std: Dict[str, List] = {k: [] for k in atmos_variables}
    atmos_transforms = {}
    for k in atmos_variables:
        for level in sorted(atmos_levels[k]):
            atmos_mean[k].append(normalize_mean[f"{k}_{level}"])
            atmos_std[k].append(normalize_std[f"{k}_{level}"])

        atmos_mean[k] = np.array(atmos_mean[k])
        atmos_std[k] = np.array(atmos_std[k])
        atmos_transforms[k] = transforms.Normalize(atmos_mean[k], atmos_std[k])

    return single_transforms, atmos_transforms


class MultiSourceDataModule(LightningDataModule):
    def __init__(
        self,
        dict_root_dirs: dict,
        dict_data_spatial_shapes: dict,
        dict_single_vars: dict,
        dict_atmos_vars: dict,
        dict_hrs_each_step: dict,
        dict_max_predict_range: dict,
        batch_size: int,
        dict_metadata_dirs: Optional[dict] = None,
        prefetch: int = 0,
        shuffle_buffer_size: int = int(1e6),
        val_shuffle_buffer_size: int = int(1e6),
        num_workers: int = 0,
        pin_memory: bool = False,
        use_old_loader: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dict_root_dirs = dict_root_dirs
        self.dict_metadata_dirs = dict_metadata_dirs
        self.dict_data_spatial_shapes = dict_data_spatial_shapes
        self.dict_single_vars = dict_single_vars
        self.dict_atmos_vars = dict_atmos_vars
        self.dict_hrs_each_step = dict_hrs_each_step
        self.dict_max_predict_range = dict_max_predict_range
        self.use_old_loader = use_old_loader
        self.train_datapipe, self.val_datapipe, self.test_datapipe = None, None, None

    def _build_datapipe(self, dict_root_dirs: dict[str, str], file_shuffle_buffer_size=None):
        records = []
        for k in dict_root_dirs.keys():
            root_dir = dict_root_dirs[k]
            self.dict_data_spatial_shapes[k]
            single_vars = self.dict_single_vars[k]
            atmos_vars = self.dict_atmos_vars[k]
            hrs_each_step = self.dict_hrs_each_step[k]
            max_predict_range = self.dict_max_predict_range[k]
            # TODO: does it make sense to have separate normalization for each dataset? Or should we compute something across oour datasets for each variable?
            if self.dict_metadata_dirs is not None:
                metadata_dir = self.dict_metadata_dirs[k]
                single_transforms, atmos_transforms = get_normalization_transforms(
                    metadata_dir, single_vars, atmos_vars
                )
                lat = np.load(os.path.join(metadata_dir, "lat.npy"))
                lon = np.load(os.path.join(metadata_dir, "lon.npy"))
                if lat.shape[0] == 721:
                    lat = lat[:-1]
            else:
                single_transforms, atmos_transforms = None, None
                lat, lon = None, None
            # todo: [SHC] add transform and combination of atoms and single vars
            record: WSeqRecord = WSeqRecord.load_record(recorddir=root_dir)
            record.set_framereader_args(
                {
                    "input_features": single_vars + atmos_vars,
                    "target_features": single_vars + atmos_vars,
                    "max_pred_steps": max_predict_range,
                }
            )
            records.append(record)

        assert len(records) == 1  # For now, we only support one data set.
        return build_wdatapipe(
            records[0],
            file_shuffle_buffer_size,
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            lat=lat,
            lon=lon,
            hrs_each_step=hrs_each_step,
            batch_size=self.hparams.batch_size,
            single_transforms=single_transforms,
            atmos_transforms=atmos_transforms,
            prefetch=self.hparams.prefetch,
            mappings=[],
        )

    def setup(self, stage: Optional[str] = None):
        assert ("train" in self.dict_root_dirs and "val" in self.dict_root_dirs) or (
            "test" in self.dict_root_dirs
        ), "The data dirs of either train&val or test must be set."
        if "train" in self.dict_root_dirs:
            self.train_datapipe = self._build_datapipe(
                self.dict_root_dirs["train"], self.hparams.shuffle_buffer_size
            )
            self.val_datapipe = self._build_datapipe(
                self.dict_root_dirs["val"], self.hparams.val_shuffle_buffer_size
            )
        if "test" in self.dict_root_dirs:
            self.test_datapipe = self._build_datapipe(self.dict_root_dirs["test"])

    def _build_dataloader(self, datapipe):
        if self.use_old_loader:
            # Batch size and collate_fn are already set in the datapipe.
            return torch.utils.data.DataLoader(
                datapipe,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_size=None,
                collate_fn=None,
            )

        # Recommended way to combine multiprocessing + distributed sharding
        # https://pytorch.org/data/main/dlv2_tutorial.html#multiprocessing-distributed
        rs = MPRS(num_workers=self.hparams.num_workers)
        if torch.distributed.is_initialized():
            dist_rs = DistributedReadingService()
            rs = SequentialReadingService(dist_rs, rs)
        return DataLoader2(datapipe, reading_service=rs)

    def train_dataloader(self):
        if self.train_datapipe is not None:
            return self._build_dataloader(self.train_datapipe)
        else:
            return None

    def val_dataloader(self):
        if self.val_datapipe is not None:
            return self._build_dataloader(self.val_datapipe)
        else:
            return None

    def test_dataloader(self):
        if self.test_datapipe is not None:
            return self._build_dataloader(self.test_datapipe)
        else:
            return None
