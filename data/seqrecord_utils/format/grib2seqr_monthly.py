import os
import psutil
import click
import time
import numpy as np
import shutil

from tqdm import tqdm
from climai_global.seqrecord.wseqrecord import WSeqRecord
from climai_global.seqrecord.utils import distribute_loads
from typing import Any, Dict, Iterator, Tuple
from multiprocessing import Process
from itertools import islice

from climai_global.paths import CLIMAI_GLOBAL_DATA_ROOT
from climai_global.data.grib_utils import read_grib_with_retry, get_num_frames_in_month
from climai_global.data.variables import VariableMapping, VariableMapType


# TODO(Cris): Integrate total precipitation in the data.
# Total precipitation is special because it has a different shape than the other variables.
# "total_precipitation",


def frame_generator(
    path: str,
    grib_path: str,
    buffer_size: int,
    m: int,
    year: int,
    variable_mapper: VariableMapType,
):
    """Generates a stream of frames from the grib2 files in the given path."""
    num_frames_in_month = get_num_frames_in_month(m, year, grib_path, variable_mapper)
    print(f"{num_frames_in_month=}")

    pbar = tqdm(
        range(0, num_frames_in_month, buffer_size),
        desc=f"Formating frames in month {m}",
    )

    # Read data in chunks of buffer_size to keep the memory under control.
    for frame_start_idx in pbar:
        np_vars = {}
        frame_end_idx = min(frame_start_idx + buffer_size, num_frames_in_month)
        # TODO: this script is currently brittle if teh
        for var in variable_mapper.surface_variables:
            print(f"Collecting single var: {var} of month {m}")
            var_path = os.path.join(path, f"{var}_0.npy")
            # Perform a copy operation to avoid memory leaks.
            np_vars[var] = np.load(var_path)[frame_start_idx:frame_end_idx].copy()

        for var in variable_mapper.atmospheric_variables:
            print(f"Collecting atmospheric var: {var} of month {m} at all pressure levels...")
            for level in variable_mapper.pressure_levels:
                var_path = os.path.join(path, f"{var}_{level}.npy")
                # Perform a copy operation to avoid memory leaks.
                np_vars[f"{var}_{level}"] = np.load(var_path)[frame_start_idx:frame_end_idx].copy()

        print(f"Collected all vars for month {m} at frames {frame_start_idx}:{frame_end_idx}...")
        print(f"RAM usage (GB): {psutil.virtual_memory()[3]/1e9}")
        print(f"Memory of the np array (GB): {sum(arr.nbytes for _, arr in np_vars.items())/1e9}")
        for frame_idx in range(0, frame_end_idx - frame_start_idx):
            frame = {}
            for key in np_vars:
                frame[key] = np_vars[key][frame_idx]
            yield frame


def grib2np(month: int, year: int, config, variable_mapper: VariableMapType):
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    local_cache_dir = (
        os.path.join(config["local_cache_dir"], WSeqRecord.subfolder_name(month, 12))
        if config["local_cache_dir"] != ""
        else None
    )

    sub_wseqrecord = WSeqRecord(
        os.path.join(
            config["wseqrecord_dir"],
            WSeqRecord.subfolder_name(month - 1, 12),
        ),
        local_cache_dir=local_cache_dir,
    )

    # Generate a stream of data frames and save them to disk.
    sub_wseqrecord.put_frame(
        frame_generator=frame_generator(
            config["download_dir"],
            config["grib_dataset_dir"],
            config["buffer_size"],
            month,
            year,
            variable_mapper,
        ),
        max_recordfile_size=config["max_recordfile_size"],
    )
    print("Month", month, " finished, dumping metadata!")
    # Index the data frames in the current month and save them to disk.
    sub_wseqrecord.dump_record(rank=month - 1)


def get_grib_file_generator(
    path: str, year: int, month: int, variable_mapper: VariableMapType
) -> Iterator[Tuple[str, str]]:
    """Generates a stream of grib files in the given path."""
    for var in variable_mapper.surface_variables:
        yield (var, os.path.join(path, f"{year}", f"{month:02d}", f"{var}_0.grib"))

    for var in variable_mapper.atmospheric_variables:
        for level in variable_mapper.pressure_levels:
            yield (var, os.path.join(path, f"{year}", f"{month:02d}", f"{var}_{level}.grib"))


def download_grib(
    rank: int,
    config: dict,
    grib_generator: Iterator[Tuple[str, str]],
    variable_mapper: VariableMapType,
) -> None:
    """Downloads grib files from the given generator and convert to numpy."""
    for var, grib_file in grib_generator:
        print(f"Rank {rank:02d}: Converting to numpy {grib_file}...")
        code = variable_mapper.get_shortname(var)
        arr = read_grib_with_retry(grib_file, var_code=code, to_numpy=True)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, copy=False, nan=0.0)

        # Save numpy file to disk
        filename = os.path.basename(grib_file).replace(".grib", ".npy")
        filepath = os.path.join(config["download_dir"], filename)
        print(f"Rank {rank:02d}: Saving numpy array to {filepath}...")
        np.save(filepath, arr)


def cleanup(month: int, download_dir):
    print(f"Cleaning up month {month}...")
    # Delete the download directory.
    shutil.rmtree(download_dir)

    # Delete the blobfuse cache.
    for path, dirs, files in os.walk("/scratch/azureml/data2"):
        for f in files:
            os.unlink(os.path.join(path, f))
    for path, dirs, files in os.walk("/scratch/azureml/data1"):
        for f in files:
            os.unlink(os.path.join(path, f))


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["CMCC", "ECWMF", "ERA5", "ERA5Pangu"], case_sensitive=False),
    default="ERA5",
)
@click.option("--dataset-mount-dir", type=str, default=CLIMAI_GLOBAL_DATA_ROOT)
@click.option("--local-cache-dir", type=str, default="~/record_cache")
@click.option("--download-dir", type=str, default="~/download_dir")
@click.option("--folder", type=str, default="train3")
@click.option("--year", type=int, required=True)
@click.option("--start_month", type=click.IntRange(1, 12), default=1)
@click.option("--end_month", type=click.IntRange(1, 12), default=12)
@click.option("--num-processes", type=int, default=48)
@click.option("--buffer-size", type=int, default=250)
@click.option(
    "--max-recordfile-size",
    type=int,
    help="The maximum size of the record file in bytes",
    default=int(2e9),
)
@click.option("--test", "-t", is_flag=True, help="Running on test variables only", default=False)
def main(
    dataset: str,
    dataset_mount_dir: str,
    local_cache_dir: str,
    download_dir: str,
    folder: str,
    year: int,
    start_month: int,
    end_month: int,
    num_processes: int,
    buffer_size: int,
    max_recordfile_size: int,
    test: bool,
):
    """Generates a range of months for ERA5 data."""
    start_time = time.time()

    # get the input variables associated with the dataset
    variable_map = VariableMapping.from_name(dataset, restricted_test_variables=test)
    assert end_month >= start_month, "end_month must be greater than or equal to start_month."

    print(
        f"Config: {year=}, {start_month=}, {end_month=} {buffer_size=}, {local_cache_dir=}, {download_dir=}.\n"
    )
    config: Dict[str, Any] = {
        "num_processes": num_processes,
        "wseqrecord_dir": os.path.join(dataset_mount_dir, "era5seqrecord", folder, str(year)),
        "local_cache_dir": local_cache_dir,
        "download_dir": os.path.abspath(os.path.expanduser(download_dir)),
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
        "buffer_size": buffer_size,
        "max_recordfile_size": max_recordfile_size,
    }

    months = range(start_month, end_month + 1)
    for m in months:
        print("Generating month", m)
        os.makedirs(config["download_dir"], exist_ok=True)

        # Download grib files, convert them to numpy and store them on disk.
        grib_generator = get_grib_file_generator(config["grib_dataset_dir"], year, m, variable_map)
        total_files = len(variable_map.surface_variables) + len(
            variable_map.atmospheric_variables
        ) * len(variable_map.pressure_levels)
        dividens = distribute_loads(total_files, config["num_processes"])
        processes = []
        for i in range(config["num_processes"]):
            sub_grib_generator = islice(grib_generator, dividens[i][0], dividens[i][1])
            p = Process(
                target=download_grib,
                args=(i, config, sub_grib_generator, variable_map),
            )
            processes.append(p)
            p.start()

        for i in range(config["num_processes"]):
            processes[i].join()

        # Convert numpy files to WSeqRecord.
        print("Converting numpy files to WSeqRecord...")
        grib2np(m, year, config, variable_map)

        # Cleaning up.
        cleanup(m, config["download_dir"])

    print(f"Total time (s): {time.time() - start_time}")


if __name__ == "__main__":
    main()
