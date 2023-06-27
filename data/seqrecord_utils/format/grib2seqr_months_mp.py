import os
import psutil
import click
import time
import numpy as np

from tqdm import tqdm
from climai_global.seqrecord.wseqrecord import WSeqRecord
from climai_global.seqrecord.utils import distribute_loads
from typing import Any, Dict, Iterator
from multiprocessing import Process
from itertools import islice
from climai_global.paths import CLIMAI_GLOBAL_DATA_ROOT
from climai_global.data.grib_utils import read_grib_with_retry, get_num_frames_in_month
from climai_global.data.variables import VariableMapping, VariableMapType

# TODO(Cris): Integrate total precipitation in the data.
# Total precipitation is special because it has a different shape than the other variables.
# "total_precipitation",


def frame_generator(
    rank: int, path, buffer_size: int, year, month_gen: Iterator, variable_mapper: VariableMapType
):
    """Generates a stream of frames from the grib2 files in the given path."""
    for m in month_gen:
        num_frames_in_month = get_num_frames_in_month(m, year, path, variable_mapper)
        print(f"{num_frames_in_month=}")

        pbar = (
            tqdm(
                range(0, num_frames_in_month, buffer_size),
                desc=f"Formating frames in month {m}",
            )
            if rank == 0
            else range(0, num_frames_in_month, buffer_size)
        )

        # Read data in chunks of buffer_size to keep the memory under control.
        for frame_start_idx in pbar:
            np_vars = {}
            frame_end_idx = min(frame_start_idx + buffer_size, num_frames_in_month)
            for var in variable_mapper.surface_variables:
                print(f"Collecting single var: {var} of month {m}")
                var_path = os.path.join(path, f"{year}", f"{m:02d}", f"{var}_0.grib")
                code = variable_mapper.get_shortname(var)
                ds = read_grib_with_retry(var_path)

                assert len(ds[code].shape) == 3

                arr = ds[code][frame_start_idx:frame_end_idx].to_numpy()
                # TODO(Cris): Variables like geopotential have NaN values. Improve how we handle them.
                if np.isnan(arr).any():
                    arr = np.nan_to_num(arr, copy=False, nan=0.0)
                np_vars[var] = arr

                del ds

            for var in variable_mapper.atmospheric_variables:
                print(f"Collecting atmospheric var: {var} of month {m} at all pressure levels...")
                for level in variable_mapper.pressure_levels:
                    var_path = os.path.join(
                        path,
                        f"{year}",
                        f"{m:02d}",
                        f"{var}_{level}.grib",
                    )
                    ds = read_grib_with_retry(var_path)
                    code = variable_mapper.get_shortname(var)
                    arr = ds[code][frame_start_idx:frame_end_idx].to_numpy()

                    # TODO(Cris): Variables like geopotential have NaN values. Improve how we handle them.
                    if np.isnan(arr).any():
                        arr = np.nan_to_num(arr, copy=False, nan=0.0)
                    np_vars[f"{var}_{level}"] = arr
                    del ds

            print(
                f"Collected all vars for month {m} at frames {frame_start_idx}:{frame_end_idx}..."
            )
            if rank == 0:
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


def grib2np(rank, world_size, config, year, month_gen, variable_mapper: VariableMapType):
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
            config["grib_dataset_dir"],
            config["buffer_size"],
            year,
            month_gen,
            variable_mapper,
        )
    )
    print("Rank", rank, " finished, dumping metadata!")
    # Index the data frames in the current month and save them to disk.
    sub_wseqrecord.dump_record(rank=rank)


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["CMCC", "ECWMF", "ERA5"], case_sensitive=False),
    default="ERA5",
)
@click.option("--dataset-mount-dir", type=str, default=CLIMAI_GLOBAL_DATA_ROOT)
@click.option("--local-cache-dir", type=str, default="~/record_cache")
@click.option("--year", type=int, required=True)
@click.option("--num-processes", type=int, default=12)
@click.option(
    "--buffer-size", type=int, default=20
)  # 20 is the maximum that can fit on a 480GB machine without problems.
@click.option("--test", "-t", is_flag=True, help="Running on test variables only", default=False)
def main(
    dataset: str,
    dataset_mount_dir: str,
    local_cache_dir: str,
    year: int,
    num_processes: int,
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
        "num_processes": num_processes,
        "wseqrecord_dir": os.path.join(
            dataset_mount_dir,
            f"era5seqrecord/train/{year}",
        ),
        "local_cache_dir": local_cache_dir,
        "grib_dataset_dir": os.path.join(dataset_mount_dir, "era5"),
        "buffer_size": buffer_size,
    }

    month_generator = range(1, 13)
    dividens = distribute_loads(12, config["num_processes"])
    processes = []
    for i in range(config["num_processes"]):
        sub_month_generator = islice(month_generator, dividens[i][0], dividens[i][1])
        p = Process(
            target=grib2np,
            args=(i, config["num_processes"], config, year, sub_month_generator, variable_map),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish.
    for i in range(config["num_processes"]):
        processes[i].join()

    print(f"Total time (s): {time.time() - start_time}")


if __name__ == "__main__":
    main()
