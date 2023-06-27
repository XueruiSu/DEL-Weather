import os
import click

from climai_global.seqrecord.wseqrecord import WSeqRecord
from climai_global.paths import CLIMAI_GLOBAL_DATA_ROOT


@click.command()
@click.option("--container", type=str, default="era5seqrecord")
@click.option("--folder", type=str, default="test")
@click.option("--year", type=int, required=True)
@click.option("--world-size", type=int, default=12)
def main(container: str, folder: str, year: int, world_size: int):
    """Indexes a seqrecord dataset consisting of a single year's data.
    This should be used for single-year training only.
    """
    recorddir = os.path.join(CLIMAI_GLOBAL_DATA_ROOT, container, folder, str(year))
    print(f"Gathering the data for year {year}...")
    record: WSeqRecord = WSeqRecord.gather_subseqrecords(recorddir, world_size=world_size)
    print("Dumping records...")
    record.dump_record()


if __name__ == "__main__":
    main()
