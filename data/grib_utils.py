import os
import time
import xarray as xr
import numpy as np

from typing import Union, Dict, Optional
from climai_global.data.variables import VariableMapType


def read_grib_with_retry(
    grib_path: str, var_code: Optional[str] = None, to_numpy: bool = False, max_retries: int = 5
) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """Reads a grib file with retries. Useful when reading data from a network drive."""
    retry_count = 0
    backoff_time = 1

    # Loop until the file is read successfully or the max retries are reached.
    while True:
        try:
            # Try to open and read the file.
            ds = xr.open_dataset(grib_path, engine="cfgrib")
            if to_numpy:
                # Numpy conversion can also raise an OSError.
                return ds[var_code].to_numpy()
            # Return the content if no exception is raised.
            return ds
        except Exception as e:
            # Print the exception and increment the retry count.
            print(f"Exception occurred: {e}")
            retry_count += 1

            # Check if the max retries are reached.
            if retry_count > max_retries:
                # Raise the exception and exit the loop.
                raise e
            else:
                # Print the retry count and the backoff time.
                print(f"Retrying {retry_count} of {max_retries} after {backoff_time} seconds")
                # Wait for the backoff time
                time.sleep(backoff_time)
                # Double the backoff time for the next retry.
                backoff_time *= 2


def get_num_frames_in_month(
    month: int, year: int, path: str, variable_mapper: VariableMapType
) -> int:
    """Returns the number of frames in the given month."""
    var = list(variable_mapper.surface_variables.keys())[0]
    var_path = os.path.join(path, f"{year}", f"{month:02d}", f"{var}_0.grib")
    ds = read_grib_with_retry(var_path)
    code = variable_mapper.get_shortname(var)
    return ds[code].shape[0]
