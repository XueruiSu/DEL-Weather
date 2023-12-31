# trainer 
# 改年份需要改两个地方：
# 1.year变量（数据路径root_dirs和metadata_dirs都用year变量赋值）
# 2.filename变量中的年份。
# 每次train之前要确认default_root_dir是否是你的code的目录
import os  

year = "1980"
os.makedirs(f"checkpoints_{year}", exist_ok=True)  
default_root_dir = "/blob/weathers2/xuerui/Dual-Weather/project/DEL-Weather"
accelerator = "gpu"
default_checkpoints_dir = default_root_dir + f"/checkpoints_{year}"
precision = 16
max_epochs = 100
# strategy = "ddp_find_unused_parameters_false"
strategy = "ddp"
devices = "auto"
# devices = 8
num_nodes = 1
enable_checkpointing = True

# callbacks
# pytorch_lightning.callbacks.ModelCheckpoint
dirpath = default_checkpoints_dir
monitor_param = "val/loss" # name of the logged metric which determines when model is improving
mode = "min" # "max" means higher metric value is better, can be also "min"
save_top_k = 2 # save k best models (determined by above metric)
save_last = True # additionaly always save model from last epoch
verbose = False  
filename = "epoch_1980_{epoch}-{step}-{val/loss:.2f}"
auto_insert_metric_name = False
# pytorch_lightning.callbacks.LearningRateMonitor
logging_interval = "step"
# pytorch_lightning.callbacks.RichModelSummary
max_depth = -1

# trainer batches:
limit_train_batches = 1.0
limit_val_batches = 160
limit_test_batches = 160

# model hyperpapremeters
patch_size=16
embed_dim=768
depth=12
decoder_depth=2
num_heads=12
mlp_ratio=4
drop_path=0.1
drop_rate=0.1
use_flash_attn=True

embed_dim_UQ=768
out_embed_dim_UQ = [256, 191]
depth_UQ=1
en_UQ_embed_dim = 256
num_heads_UQ_en=1
num_heads_UQ_down=4
mlp_ratio_UQ=4
drop_path_UQ=0.1
drop_rate_UQ=0.1
use_flash_attn_UQ=True
decoder_depth_down_UQ = 3

# checkpoints Main Model & UQ:
Main_Model_Path_1 = default_root_dir + "/checkpoints_UQ/"
Main_Model_Path_2 = default_root_dir + "/checkpoints_UQ/"
UQ_Model_path = default_root_dir + "/checkpoints/"
# UQ model optimization hyperparameters
lr_UQ = 5e-4
weight_decay_UQ = 1e-4
beta_1_UQ = 0.9
beta_2_UQ = 0.95
T_max_UQ = 10

const_vars = [
    "land_sea_mask",
    "orography",
    "lattitude",
]
single_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_column_water_vapour",
    # "toa_incident_solar_radiation",
    # "total_precipitation",
]
atmos_levels = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
atmos_vars = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    # "relative_humidity",
]


# ForecastPretrain param
restart_path=""
lr = 5e-4
beta_1 = 0.9
beta_2 = 0.95
weight_decay = 1e-5
warmup_steps = 1000
warmup_start_lr = 1e-4
eta_min = 1e-05
opt_name = "adamw"


# dataloader hyperparemeters
dict_root_dirs = {
    "train": {
        # "era5":"/mnt/data/era5/1979/",
        # "era5":"/mnt/data/era5_second/1980/",
        "era5": f"/nfs/weather/era5/{year}",
        # "era5":"/nfs/weather/era5/1980",
        # "era5":"/blob/weathers2/xuerui/Dual-Weather/data/era5/1979",
    },
    "val": {
        "era5":"/nfs/weather/era5_valid/1982/",
        # "era5":"/blob/weathers2/xuerui/Dual-Weather/data/era5/1979/",
    },
    "test": {
        "era5":"/nfs/weather/era5_test/1981/",
        # "era5":"/blob/weathers2/xuerui/Dual-Weather/data/era5/1979/",
    }
}
dict_metadata_dirs = {
    # "era5":"/mnt/data/era5/1979/",
    # "era5":"/mnt/data/era5_second/1980/",
    "era5": default_root_dir + f"/data/era5_data_utils/era5_{year}",
    # "era5":"/blob/weathers2/xuerui/Dual-Weather/project/DEL-Weather/data/era5_data_utils/era5_1980",
    # "era5":"/blob/weathers2/xuerui/Dual-Weather/data/era5_second/1980/",
}
dict_data_spatial_shapes = {"era5": [721, 1440],}
dict_single_vars = {
    "era5": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "total_column_water_vapour",
    ]
}
dict_atmos_vars = {
    "era5":[
            'geopotential_1',
            'geopotential_2',
            'geopotential_3',
            'geopotential_5',
            'geopotential_7',
            'geopotential_10',
            'geopotential_20',
            'geopotential_30',
            'geopotential_50',
            'geopotential_70',
            'geopotential_100',
            'geopotential_125',
            'geopotential_150',
            'geopotential_175',
            'geopotential_200',
            'geopotential_225',
            'geopotential_250',
            'geopotential_300',
            'geopotential_350',
            'geopotential_400',
            'geopotential_450',
            'geopotential_500',
            'geopotential_550',
            'geopotential_600',
            'geopotential_650',
            'geopotential_700',
            'geopotential_750',
            'geopotential_775',
            'geopotential_800',
            'geopotential_825',
            'geopotential_850',
            'geopotential_875',
            'geopotential_900',
            'geopotential_925',
            'geopotential_950',
            'geopotential_975',
            'geopotential_1000',
            'u_component_of_wind_1',
            'u_component_of_wind_2',
            'u_component_of_wind_3',
            'u_component_of_wind_5',
            'u_component_of_wind_7',
            'u_component_of_wind_10',
            'u_component_of_wind_20',
            'u_component_of_wind_30',
            'u_component_of_wind_50',
            'u_component_of_wind_70',
            'u_component_of_wind_100',
            'u_component_of_wind_125',
            'u_component_of_wind_150',
            'u_component_of_wind_175',
            'u_component_of_wind_200',
            'u_component_of_wind_225',
            'u_component_of_wind_250',
            'u_component_of_wind_300',
            'u_component_of_wind_350',
            'u_component_of_wind_400',
            'u_component_of_wind_450',
            'u_component_of_wind_500',
            'u_component_of_wind_550',
            'u_component_of_wind_600',
            'u_component_of_wind_650',
            'u_component_of_wind_700',
            'u_component_of_wind_750',
            'u_component_of_wind_775',
            'u_component_of_wind_800',
            'u_component_of_wind_825',
            'u_component_of_wind_850',
            'u_component_of_wind_875',
            'u_component_of_wind_900',
            'u_component_of_wind_925',
            'u_component_of_wind_950',
            'u_component_of_wind_975',
            'u_component_of_wind_1000',
            'v_component_of_wind_1',
            'v_component_of_wind_2',
            'v_component_of_wind_3',
            'v_component_of_wind_5',
            'v_component_of_wind_7',
            'v_component_of_wind_10',
            'v_component_of_wind_20',
            'v_component_of_wind_30',
            'v_component_of_wind_50',
            'v_component_of_wind_70',
            'v_component_of_wind_100',
            'v_component_of_wind_125',
            'v_component_of_wind_150',
            'v_component_of_wind_175',
            'v_component_of_wind_200',
            'v_component_of_wind_225',
            'v_component_of_wind_250',
            'v_component_of_wind_300',
            'v_component_of_wind_350',
            'v_component_of_wind_400',
            'v_component_of_wind_450',
            'v_component_of_wind_500',
            'v_component_of_wind_550',
            'v_component_of_wind_600',
            'v_component_of_wind_650',
            'v_component_of_wind_700',
            'v_component_of_wind_750',
            'v_component_of_wind_775',
            'v_component_of_wind_800',
            'v_component_of_wind_825',
            'v_component_of_wind_850',
            'v_component_of_wind_875',
            'v_component_of_wind_900',
            'v_component_of_wind_925',
            'v_component_of_wind_950',
            'v_component_of_wind_975',
            'v_component_of_wind_1000',
            'temperature_1',
            'temperature_2',
            'temperature_3',
            'temperature_5',
            'temperature_7',
            'temperature_10',
            'temperature_20',
            'temperature_30',
            'temperature_50',
            'temperature_70',
            'temperature_100',
            'temperature_125',
            'temperature_150',
            'temperature_175',
            'temperature_200',
            'temperature_225',
            'temperature_250',
            'temperature_300',
            'temperature_350',
            'temperature_400',
            'temperature_450',
            'temperature_500',
            'temperature_550',
            'temperature_600',
            'temperature_650',
            'temperature_700',
            'temperature_750',
            'temperature_775',
            'temperature_800',
            'temperature_825',
            'temperature_850',
            'temperature_875',
            'temperature_900',
            'temperature_925',
            'temperature_950',
            'temperature_975',
            'temperature_1000',
            'specific_humidity_1',
            'specific_humidity_2',
            'specific_humidity_3',
            'specific_humidity_5',
            'specific_humidity_7',
            'specific_humidity_10',
            'specific_humidity_20',
            'specific_humidity_30',
            'specific_humidity_50',
            'specific_humidity_70',
            'specific_humidity_100',
            'specific_humidity_125',
            'specific_humidity_150',
            'specific_humidity_175',
            'specific_humidity_200',
            'specific_humidity_225',
            'specific_humidity_250',
            'specific_humidity_300',
            'specific_humidity_350',
            'specific_humidity_400',
            'specific_humidity_450',
            'specific_humidity_500',
            'specific_humidity_550',
            'specific_humidity_600',
            'specific_humidity_650',
            'specific_humidity_700',
            'specific_humidity_750',
            'specific_humidity_775',
            'specific_humidity_800',
            'specific_humidity_825',
            'specific_humidity_850',
            'specific_humidity_875',
            'specific_humidity_900',
            'specific_humidity_925',
            'specific_humidity_950',
            'specific_humidity_975',
            'specific_humidity_1000',],
}
dict_hrs_each_step = {
    "era5": 1,  
}
dict_max_predict_range = {
    "era5": 1,
}
batch_size = 2
shuffle_buffer_size = 10
val_shuffle_buffer_size = 10
# 这里调成大于0的数总是会报设备内存不够的错误
num_workers = 0
pin_memory = True
use_old_loader = False







