# src: how to run

No compilation. Python 3.8+ (tested on 3.12), standard library only, single thread.

```bash
python main.py <path/to/imu0/data.csv | path/to/folder_containing_imu0> \
               [-o estimated_trajectory.txt] [--blind 5.0] [--metadata metadata.json]
```

* Reads the EuRoC IMU csv by streaming (constant memory); tolerant of spaces after commas, CRLF,
  NaN or garbage rows, duplicate or out-of-order timestamps.
* Writes `estimated_trajectory.txt` in TUM format: `t[s] tx ty tz qx qy qz qw`, one pose per input row
  plus `--blind` seconds of dead-reckoned poses at the nominal IMU period.
* `--metadata` writes runtime and peak-RSS numbers (optional bundle file).

Note: `output/estimated_trajectory.txt` in the repo was produced from a *synthetic* test flight.
Regenerate it from the organisers' degraded `imu0/data.csv` before final submission.
