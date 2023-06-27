import os
import psutil
import click
import time
import numpy as np
import shutil

from tqdm import tqdm
from climai_global.seqrecord.wseqrecord import WSeqRecord
from typing import Any, Dict
from multiprocessing import Process
from climai_global.paths import CLIMAI_GLOBAL_DATA_ROOT
from climai_global.data.grib_utils import read_grib_with_retry, get_num_frames_in_month
from climai_global.data.variables import VariableMapping, VariableMapType

# TODO(Cris): Integrate total precipitation in the data.
# Total precipitation is special because it has a different shape than the other variables.
# "total_precipitation",


def delete_blobfuse_read_cache(month: int):
    """Deletes the blobfuse cache for the specified month."""
    print(f"Month {month:02d}: Deleting blobfuse cache...")
    # This is where amlt blobfuse cache is stored by default.
    for path, dirs, files in os.walk("/scratch/azureml/data2"):
        for f in files:
            # Avoid deleting files used by other processes.
            if f"{month:02d}" in path:
                os.unlink(os.path.join(path, f))


def read_var(
    rank: int,
    var: str,
    var_path: str,
    grib_dataset_dir: str,
    year,
    m: int,
    variable_mapper: VariableMapType,
):
    # We already cached the file, so we can load from disk.
    if os.path.exists(var_path):
        return np.load(var_path)

    # Download the grib file and save to disk.
    grib_file = os.path.basename(var_path).replace(".npy", ".grib")
    grib_path = os.path.join(grib_dataset_dir, f"{year}", f"{m:02d}", grib_file)
    print(f"Rank {rank:02d}: Loading {var} from {grib_path}..")

    code = variable_mapper.get_shortname(var)
    arr = read_grib_with_retry(grib_path, var_code=code, to_numpy=True)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, copy=False, nan=0.0)

    # Save numpy file to disk
    print(f"Rank {rank:02d}: Saving numpy array to {var_path}...")
    os.makedirs(os.path.dirname(var_path), exist_ok=True)
    np.save(var_path, arr)

    return arr


def print_logs(rank, var, np_vars):
    if rank % 3 == 0:
        print(f"RAM usage (GB) before var {var} {psutil.virtual_memory()[3]/1e9}")
        print(
            f"Memory of the np array (GB) before var {var}: {sum(arr.nbytes for _, arr in np_vars.items())/1e9}"
        )


def frame_generator(
    rank: int,
    download_dir: str,
    grib_dir: str,
    buffer_size: int,
    year: int,
    variable_mapper: VariableMapType,
):
    """Generates a stream of frames from the grib2 files in the given path."""
    m = rank + 1
    num_frames_in_month = get_num_frames_in_month(m, year, grib_dir, variable_mapper)
    print(f"{num_frames_in_month=}")

    pbar = (
        tqdm(
            range(0, num_frames_in_month, buffer_size),
            desc=f"Formating frames in month {m}",
        )
        if rank % 3 == 0
        else range(0, num_frames_in_month, buffer_size)
    )

    # Read data in chunks of buffer_size to keep the memory under control.
    for frame_start_idx in pbar:
        np_vars: dict[str, np.ndarray] = {}
        frame_end_idx = min(frame_start_idx + buffer_size, num_frames_in_month)
        for var in variable_mapper.surface_variables:
            print(f"Collecting single var: {var} of month {m}")
            print_logs(rank, var, np_vars)

            var_path = os.path.join(download_dir, f"{m:02d}", f"{var}_0.npy")
            arr = read_var(rank, var, var_path, grib_dir, year, m, variable_mapper)
            # We need to perform a copy here to avoid a memory leak.
            np_vars[var] = arr[frame_start_idx:frame_end_idx].copy()
            del arr

        delete_blobfuse_read_cache(rank)

        for var in variable_mapper.atmospheric_variables:
            print(f"Collecting atmospheric var: {var} of month {m} at all pressure levels...")
            print_logs(rank, var, np_vars)

            for level in variable_mapper.pressure_levels:
                var_path = os.path.join(
                    download_dir,
                    f"{m:02d}",
                    f"{var}_{level}.npy",
                )
                arr = read_var(rank, var, var_path, grib_dir, year, m, variable_mapper)
                # We need to perform a copy here to avoid a memory leak.
                np_vars[f"{var}_{level}"] = arr[frame_start_idx:frame_end_idx].copy()
                del arr

            delete_blobfuse_read_cache(rank)

        print(f"Collected all vars for month {m} at frames {frame_start_idx}:{frame_end_idx}...")
        if rank % 3 == 0:
            print(f"RAM usage (GB): {psutil.virtual_memory()[3]/1e9}")
            print(
                f"Memory of the np array (GB): {sum(arr.nbytes for _, arr in np_vars.items())/1e9}"
            )
        # TODO(Cris): Investigate why processes crash at this stage when using very large frame sizes.
        for frame_idx in range(0, frame_end_idx - frame_start_idx):
            frame = {}
            for key in np_vars:
                frame[key] = np_vars[key][frame_idx]
            yield frame


def grib2seqr(
    rank: int, world_size: int, config: Dict[str, Any], year: int, variable_mapper: VariableMapType
) -> None:
    """
    Convert grib files to numpy arrays and save them to disk.
    """
    local_cache_dir = (
        os.path.join(config["local_cache_dir"], WSeqRecord.subfolder_name(rank, world_size))
        if config["local_cache_dir"] != ""
        else None
    )
    sub_wseqrecord = WSeqRecord(
        os.path.join(
            config["wseqrecord_dir"],
            WSeqRecord.subfolder_name(rank, world_size),
        ),
        # TODO(Cris): Investigate why jobs using caching seem to be slower than those without.
        local_cache_dir=local_cache_dir,
    )

    # Generate a stream of data frames and save them to disk.
    sub_wseqrecord.put_frame(
        frame_generator(
            rank,
            config["download_dir"],
            config["grib_dataset_dir"],
            config["buffer_size"],
            year,
            variable_mapper,
        )
    )

    print("Rank", rank, " finished, dumping metadata...")
    # Index the data frames in the current month and save them to disk.
    sub_wseqrecord.dump_record(rank=rank)


def generate_quarter(
    quarter: int, year: int, config: Dict[str, Any], variable_mapper: VariableMapType
):
    """Generates data for 3 months. Months are done in parallel."""
    print(f"Generating data for Q{quarter+1} of year {year}...")
    start_time = time.time()

    month_generator = range(3 * quarter + 1, 3 * (quarter + 1) + 1)
    processes = []
    for month in month_generator:
        p = Process(
            target=grib2seqr,
            args=(month - 1, 12, config, year, variable_mapper),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish.
    for i in range(len(month_generator)):
        processes[i].join()

    # Cleanup the data saved to disk in this quarter
    print(f"Cleaning up Q{quarter+1} of year {year}...")
    for month in month_generator:
        shutil.rmtree(os.path.join(config["download_dir"], f"{month:02d}"))

    print(f"Total time (s) to generate Q{quarter+1}: {time.time() - start_time}")


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["CMCC", "ECWMF", "ERA5"], case_sensitive=False),
    default="ERA5",
)
@click.option("--dataset-mount-dir", type=str, default=CLIMAI_GLOBAL_DATA_ROOT)
@click.option("--local-cache-dir", type=str, default="~/record_cache")
@click.option("--download-dir", type=str, default="~/download_dir")
@click.option("--folder", type=str, default="train")
@click.option("--year", type=int, required=True)
@click.option("--buffer-size", type=int, default=60)
@click.option("--test", "-t", is_flag=True, help="Running on test variables only", default=False)
def main(
    dataset: str,
    dataset_mount_dir: str,
    local_cache_dir: str,
    download_dir: str,
    folder: str,
    year: int,
    buffer_size: int,
    test: bool,
):
    """Generate a year of WSeqRecord data by reading data in small chunks to keep memory usage low.
    Each month must be generated by exactly one process to avoid join between processes.
    """
    start_time = time.time()

    # get the input variables associated with the dataset
    variable_map = VariableMapping.from_name(dataset, restricted_test_variables=test)

    print(f"Config: {year=}, {buffer_size=}, {local_cache_dir=}.\n")
    config: Dict[str, Any] = {
        "wseqrecord_dir": os.path.join(
            dataset_mount_dir,
            "era5seqrecord",
            folder,
            str(year),
        ),
        "local_cache_dir": local_cache_dir,
        "download_dir": os.path.abspath(os.path.expanduser(download_dir)),
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
        "buffer_size": buffer_size,
    }

    # We can fit one quarter of ERA5 data on disk.
    # Generate quarters sequentially and the months in each quarter in parallel.
    for i in range(4):
        generate_quarter(i, year, config, variable_map)

    print(f"Total time (s): {time.time() - start_time}")


if __name__ == "__main__":
    main()
