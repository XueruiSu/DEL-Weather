import os
import click

from climai_global.seqrecord.wseqrecord import WSeqRecord
from climai_global.paths import CLIMAI_GLOBAL_DATA_ROOT
from climai_global.seqrecord.utils import distribute_loads
from itertools import islice
from multiprocessing import Process
from typing import Iterator


def index_years(rank: int, recorddir: str, year_generator: Iterator[int], world_size: int):
    """
    Indexes the data of years in year_generator.
    """
    for year in year_generator:
        print(f"Rank {rank:02d}: Gathering {year}'s data")
        sub_dir = os.path.join(recorddir, str(year))
        subrecord: WSeqRecord = WSeqRecord.gather_subseqrecords(sub_dir, world_size)
        print(f"Rank {rank:02d}: Dumping records for year {year}...")
        subrecord.dump_record(rank=year - 1979)
        print(f"Rank {rank:02d}: Year {year} finished!")


@click.command()
@click.option("--container", type=str, default="era5seqrecord")
@click.option("--folder", type=str, default="test")
@click.option("--start_year", type=int, required=True)
@click.option("--end_year", type=int, required=True)
@click.option("--num_processes", type=int, default=10)
@click.option("--world-size", type=int, default=12)
@click.option("--only-global", is_flag=True, help="Skip the indexing of the individual years.")
def main(
    container: str,
    folder: str,
    start_year: int,
    end_year: int,
    num_processes: int,
    world_size: int,
    only_global: bool,
):
    """
    Indexes a seqrecord dataset with tree structure recorddir/year/month.
    This should be run after the seqrecord dataset is created.
    Each process indexes a subsequence of years.
    """
    year_generator = range(start_year, end_year + 1)
    num_years = len(year_generator)
    assert (
        num_years >= num_processes
    ), f"Number of processes ({num_processes}) > number of years ({num_years})."
    recorddir = os.path.join(CLIMAI_GLOBAL_DATA_ROOT, container, folder)

    if not only_global:
        dividens = distribute_loads(num_years, num_processes)
        processes = []
        for i in range(num_processes):
            sub_year_generator = islice(year_generator, dividens[i][0], dividens[i][1])
            p = Process(
                target=index_years,
                args=(i, recorddir, sub_year_generator, world_size),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to finish.
        for i in range(num_processes):
            processes[i].join()

    # Index the data of all the years at recorddir/record_all.dict
    print("Gathering all years' data")
    record: WSeqRecord = WSeqRecord.gather_subseqrecords(
        recorddir,
        num_years,
        rank2folder={i: str(year) for i, year in enumerate(year_generator)},
    )
    print("Dumping records for all years...")
    record.dump_record()


if __name__ == "__main__":
    main()
