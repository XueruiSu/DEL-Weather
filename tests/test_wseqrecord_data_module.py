from data.seqrecord.module import MultiSourceDataModule
import os
import torch
import pytest
from configs.Climax_train_modelparam import * # hyperparameters

DICT_ATMOS_VARS = {
    "era5": [
        "geopotential_1",
        "geopotential_2",
        "geopotential_3",
        "geopotential_5",
        "geopotential_7",
        "geopotential_10",
        "geopotential_20",
        "geopotential_30",
        "geopotential_50",
        "geopotential_70",
        "geopotential_100",
        "geopotential_125",
        "geopotential_150",
        "geopotential_175",
        "geopotential_200",
        "geopotential_225",
        "geopotential_250",
        "geopotential_300",
        "geopotential_350",
        "geopotential_400",
        "geopotential_450",
        "geopotential_500",
        "geopotential_550",
        "geopotential_600",
        "geopotential_650",
        "geopotential_700",
        "geopotential_750",
        "geopotential_775",
        "geopotential_800",
        "geopotential_825",
        "geopotential_850",
        "geopotential_875",
        "geopotential_900",
        "geopotential_925",
        "geopotential_950",
        "geopotential_975",
        "geopotential_1000",
        "u_component_of_wind_1",
        "u_component_of_wind_2",
        "u_component_of_wind_3",
        "u_component_of_wind_5",
        "u_component_of_wind_7",
        "u_component_of_wind_10",
        "u_component_of_wind_20",
        "u_component_of_wind_30",
        "u_component_of_wind_50",
        "u_component_of_wind_70",
        "u_component_of_wind_100",
        "u_component_of_wind_125",
        "u_component_of_wind_150",
        "u_component_of_wind_175",
        "u_component_of_wind_200",
        "u_component_of_wind_225",
        "u_component_of_wind_250",
        "u_component_of_wind_300",
        "u_component_of_wind_350",
        "u_component_of_wind_400",
        "u_component_of_wind_450",
        "u_component_of_wind_500",
        "u_component_of_wind_550",
        "u_component_of_wind_600",
        "u_component_of_wind_650",
        "u_component_of_wind_700",
        "u_component_of_wind_750",
        "u_component_of_wind_775",
        "u_component_of_wind_800",
        "u_component_of_wind_825",
        "u_component_of_wind_850",
        "u_component_of_wind_875",
        "u_component_of_wind_900",
        "u_component_of_wind_925",
        "u_component_of_wind_950",
        "u_component_of_wind_975",
        "u_component_of_wind_1000",
        "v_component_of_wind_1",
        "v_component_of_wind_2",
        "v_component_of_wind_3",
        "v_component_of_wind_5",
        "v_component_of_wind_7",
        "v_component_of_wind_10",
        "v_component_of_wind_20",
        "v_component_of_wind_30",
        "v_component_of_wind_50",
        "v_component_of_wind_70",
        "v_component_of_wind_100",
        "v_component_of_wind_125",
        "v_component_of_wind_150",
        "v_component_of_wind_175",
        "v_component_of_wind_200",
        "v_component_of_wind_225",
        "v_component_of_wind_250",
        "v_component_of_wind_300",
        "v_component_of_wind_350",
        "v_component_of_wind_400",
        "v_component_of_wind_450",
        "v_component_of_wind_500",
        "v_component_of_wind_550",
        "v_component_of_wind_600",
        "v_component_of_wind_650",
        "v_component_of_wind_700",
        "v_component_of_wind_750",
        "v_component_of_wind_775",
        "v_component_of_wind_800",
        "v_component_of_wind_825",
        "v_component_of_wind_850",
        "v_component_of_wind_875",
        "v_component_of_wind_900",
        "v_component_of_wind_925",
        "v_component_of_wind_950",
        "v_component_of_wind_975",
        "v_component_of_wind_1000",
        "temperature_1",
        "temperature_2",
        "temperature_3",
        "temperature_5",
        "temperature_7",
        "temperature_10",
        "temperature_20",
        "temperature_30",
        "temperature_50",
        "temperature_70",
        "temperature_100",
        "temperature_125",
        "temperature_150",
        "temperature_175",
        "temperature_200",
        "temperature_225",
        "temperature_250",
        "temperature_300",
        "temperature_350",
        "temperature_400",
        "temperature_450",
        "temperature_500",
        "temperature_550",
        "temperature_600",
        "temperature_650",
        "temperature_700",
        "temperature_750",
        "temperature_775",
        "temperature_800",
        "temperature_825",
        "temperature_850",
        "temperature_875",
        "temperature_900",
        "temperature_925",
        "temperature_950",
        "temperature_975",
        "temperature_1000",
        "specific_humidity_1",
        "specific_humidity_2",
        "specific_humidity_3",
        "specific_humidity_5",
        "specific_humidity_7",
        "specific_humidity_10",
        "specific_humidity_20",
        "specific_humidity_30",
        "specific_humidity_50",
        "specific_humidity_70",
        "specific_humidity_100",
        "specific_humidity_125",
        "specific_humidity_150",
        "specific_humidity_175",
        "specific_humidity_200",
        "specific_humidity_225",
        "specific_humidity_250",
        "specific_humidity_300",
        "specific_humidity_350",
        "specific_humidity_400",
        "specific_humidity_450",
        "specific_humidity_500",
        "specific_humidity_550",
        "specific_humidity_600",
        "specific_humidity_650",
        "specific_humidity_700",
        "specific_humidity_750",
        "specific_humidity_775",
        "specific_humidity_800",
        "specific_humidity_825",
        "specific_humidity_850",
        "specific_humidity_875",
        "specific_humidity_900",
        "specific_humidity_925",
        "specific_humidity_950",
        "specific_humidity_975",
        "specific_humidity_1000",
    ],
}


# TODO(Cris): This takes about 3 min. Can we make it faster?
#             This would also improve our iteration time when developing.
@pytest.mark.with_data(reason="Sort out CI blob storage connection")
def test_seqrecord_multisource_datamodule():
    # dict_root_dirs = {
    #     "era5": os.path.join("/blob/weathers2/xuerui/Dual-Weather/data/era5/1979"),
    # }
    # dict_data_spatial_shapes = {
    #     "era5": [721, 1440],
    # }
    # dict_single_vars = {
    #     "era5": [
    #         "2m_temperature",
    #         "10m_u_component_of_wind",
    #         "10m_v_component_of_wind",
    #         "mean_sea_level_pressure",
    #         "total_cloud_cover",
    #         "total_column_water_vapour",
    #     ]
    # }

    # dict_atmos_vars = DICT_ATMOS_VARS
    # dict_hrs_each_step = {"era5": 1}
    # dict_max_predict_range = {"era5": 384}
    # datamodule = MultiSourceDataModule(
    #     dict_root_dirs,
    #     dict_data_spatial_shapes=dict_data_spatial_shapes,
    #     dict_single_vars=dict_single_vars,
    #     dict_atmos_vars=dict_atmos_vars,
    #     dict_hrs_each_step=dict_hrs_each_step,
    #     dict_max_predict_range=dict_max_predict_range,
    #     batch_size=batch_size,
    #     prefetch=0,
    #     num_workers=0,
    #     pin_memory=False,
    #     use_data_buffer=True,
    # )
    datamodule = MultiSourceDataModule(dict_root_dirs, dict_data_spatial_shapes, 
                                             dict_single_vars, dict_atmos_vars, dict_hrs_each_step, 
                                             dict_max_predict_range, batch_size, dict_metadata_dirs, 
                                             shuffle_buffer_size=shuffle_buffer_size, 
                                             val_shuffle_buffer_size=val_shuffle_buffer_size, 
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             use_old_loader=use_old_loader)
    datamodule.setup()
    tdl = datamodule.train_dataloader()

    idx = 0
    for single_inputs, atmos_inputs, single_outputs, atmos_outputs, lead_times, metadata in tdl:
        print("single_inputs", single_inputs.shape)
        for k in atmos_inputs:
            print("atmos_inputs", k, atmos_inputs[k].shape)
        print("single_outputs", single_outputs.shape)
        for k in atmos_outputs:
            print("atmos_outputs", k, atmos_outputs[k].shape)
        print("lead_times", lead_times.shape)
        print("metadata lat", metadata.lat.shape)
        print("metadata lon", metadata.lon.shape)
        assert (
            single_inputs.shape[0] == single_outputs.shape[0] == lead_times.shape[0] == batch_size
        )
        assert not torch.isnan(single_inputs).any()

        for k in atmos_inputs:
            assert atmos_inputs[k].shape[0] == atmos_outputs[k].shape[0] == batch_size
            assert (
                atmos_inputs[k].shape[-2:]
                == atmos_outputs[k].shape[-2:]
                == (metadata.lat.shape[0], metadata.lon.shape[0])
            )
            assert atmos_inputs[k].squeeze(1).shape == atmos_outputs[k].shape
            assert not torch.isnan(atmos_inputs[k]).any()

        assert lead_times.shape == (batch_size,)
        assert not torch.isnan(lead_times).any()

        if idx > 5:
            break
        idx += 1

    assert idx > 0


if __name__ == "__main__":
    test_seqrecord_multisource_datamodule()
