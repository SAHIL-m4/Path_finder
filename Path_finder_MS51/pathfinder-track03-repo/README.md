# Path Finder: Trajectory Estimation Under Sensor Degradation (Hackathon Challenge 2026, Track 03)

Standalone, single-thread, stdlib-only Python pipeline that turns a degraded EuRoC-format IMU
stream into a 6-DoF TUM trajectory, bridges dropouts, and dead-reckons the 5 s after the IMU dies.

> **Video walkthrough:** `<PASTE LINK HERE>` (also in [`video/VIDEO_LINK.md`](video/VIDEO_LINK.md))
> **Presentation:** [`presentation/Path_Finder_Track03_Presentation.pptx`](presentation/Path_Finder_Track03_Presentation.pptx)
> **Browser demo:** [`web/pathfinder.html`](web/pathfinder.html) (open the file locally, or use the hosted link: `<PASTE LINK HERE>`)
> **AI disclosure:** [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md)

## Submission checklist

| Item | Where |
|---|---|
| Source code | [`src/main.py`](src/main.py) |
| Presentation | [`presentation/`](presentation/) |
| Video | link above / [`video/VIDEO_LINK.md`](video/VIDEO_LINK.md) |
| AI disclosure | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) |
| README | this file and [`src/README.md`](src/README.md) |
| APK/SDK | N/A. Track 03 delivers a standalone Python script. A browser demo is in [`web/`](web/). |

## Run it

```bash
python src/main.py path/to/imu0/data.csv -o estimated_trajectory.txt --metadata metadata.json
# or point at the unzipped folder that contains imu0/data.csv
python src/main.py path/to/dataset_folder
```

Python 3.8+ (developed on 3.12). No third-party packages. `--blind 5.0` sets the post-failure horizon
(seconds). Input/output formats follow the challenge slides:

* **Input:** `imu0/data.csv`, `#timestamp [ns], wx, wy, wz, ax, ay, az` (accel includes gravity).
* **Output:** `estimated_trajectory.txt`, space separated, `t[s] tx ty tz qx qy qz qw`
  (unit quaternion, **xyzw**). One pose per input IMU row, then the blind-window poses at the
  nominal IMU period. Timestamps are written from the integer nanoseconds (`ns -> "%d.%09d"`),
  so there is no float rounding.

## Approach

1. **Static alignment (first 0.5 s):** mean accel gives gravity direction (roll/pitch, yaw = 0) and
   measured |g|; mean gyro gives the initial bias, but only if that window really is quiet.
2. **Attitude:** exact quaternion exponential-map integration. A Mahony-style gravity correction is
   gated on |a|~g and is weaker when measured accel noise is high. It also tracks roll/pitch gyro bias
   (clamped). The gravity feedback doubles as an implicit horizontal accel-bias absorber.
3. **Position:** rotate accel to the world frame, remove gravity, integrate twice. Zero-velocity
   updates fire only when gyro power, accel jitter, `||a|-g|` and the estimated linear acceleration
   are all small for 0.3 s. A weak, **noise-adaptive velocity leak** (stronger as the measured accel
   difference-noise grows) stops bias drift from becoming quadratic position drift.
4. **Dropouts (T3):** a gap is `dt > 2.5 x nominal`; NaN, garbage and short rows count too. Gaps are
   bridged by damped constant-rate / constant-velocity coasting. The huge `dt` is never fed to the
   integrator, and normal integration resumes with one nominal step.
5. **Blind 5 s (T2):** velocity = blend of filter velocity and the least-squares slope of the last 1 s of
   positions. Angular rate = EMA of the last ~0.3 s. Both decay exponentially (tau 8 s / 2 s).

All tunables are named constants at the top of `main.py`.

## Resource budget (measured, Linux, Python 3.12)

| Budget | Limit | Measured |
|---|---|---|
| Compute | <= 20 ms per 1 s of IMU (200 Hz) | **about 1.3 ms per 1 s of IMU** (about 6 us per sample) |
| Python heap peak (tracemalloc) | <= 8 MB | **0.8 MB** (streaming, O(1) state) |
| Process RSS peak | <= 8 MB | **about 12 MB total, of which about 9 MB is the bare CPython interpreter** |
| Threads | 1 | 1 |

The RSS row matters if the organisers measure whole-process RSS: the interpreter alone exceeds 8 MB,
so no Python submission can get below it. Our own allocation on top of it is about 3 MB. Please ask
how memory is measured (the Python heap is the meaningful number for this script).

## Validation, and what it does and does not show

There is **no real EuRoC/TUM-VI data or organiser degradation script in this repo**. The pipeline was
developed and tuned against a synthetic generator, [`tools/synth_and_eval.py`](tools/synth_and_eval.py)
(needs numpy; not part of the runtime). It simulates a 200 Hz EuRoC-like IMU (EuRoC-scale biases and
noise), then applies a progressive noise/bias ramp, a 2 s dropout, and IMU death followed by a 5 s blind
window, and scores ATE after SE(3) alignment, as `evo -a` does.

Typical result on the synthetic set (3 seeds x levels 0 / 0.5 / 1.0, 45 s flights, about 4 m amplitude):

| Segment | ATE RMSE |
|---|---|
| Pre-failure (including the 2 s dropout) | about 3 m |
| Blind 5 s | about 6 m |

Treat these as relative indicators only: absolute accuracy on the real organiser data will differ.
Pure inertial navigation from an uncalibrated MEMS IMU drifts; this pipeline limits the drift but
cannot remove it. To score on the real data with `evo`:

```bash
evo_ape tum groundtruth.tum estimated_trajectory.txt -a --t_max_diff 0.01
evo_rpe tum groundtruth.tum estimated_trajectory.txt -a -d 1 -u s
```

## Known limitations

* Tuned on synthetic data only. Real EuRoC / TUM-VI behaviour is unverified.
* Accel bias is not estimated explicitly (only absorbed by the gravity feedback and the leak).
* The velocity leak slightly biases real fast motion; it is deliberately near zero on low-noise data.
* Yaw and position are unobservable from the IMU alone, so errors grow without bound after degradation.
* One pose per *received* row. Rows removed by a dropout get no pose (no timestamp to evaluate),
  and the blind-window poses use synthetic timestamps at the median IMU period.

## Repo layout

```
src/main.py            the pipeline (entrypoint)
src/README.md          compile/run notes for the bundle
tools/synth_and_eval.py synthetic data generator + ATE/RPE scorer (numpy, dev only)
output/                sample output from the SYNTHETIC set (not real data)
presentation/          slides (built on the hackathon template)
web/pathfinder.html    in-browser demo of the same pipeline (JavaScript port)
team_name.zip          evaluation bundle (regenerate its output from the real dataset)
video/VIDEO_LINK.md    walkthrough link
AI_DISCLOSURE.md       AI usage statement
```
