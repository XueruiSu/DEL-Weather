# ERA5 data generation

To generate ERA5 data, first run the following script, specifying the year-range for which data should be generated:
```bash
bash ./projects/climai_global/scripts/era5_to_seqr.sh 1979 2016
```
This will run the `scripts/grib2seqr_buffered_mp.py` script (on the cluster) for the specified years, with one machine per year.

After the data is generated, the seqrecords must be indexed by running (from the sandbox or cluster):
```bash
python seqrecord/format/gather_seqr_years.py --start_year=1979 --end_year=2016
```
For creating a (test) dataset with only one year of weather (e.g. 1996), the data can be indexed via
```bash
python seqrecord/format/gather_seqr_year.py --year=1996
```