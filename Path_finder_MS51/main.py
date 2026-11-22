#!/usr/bin/env python3
"""
Path Finder (Track 03): 6-DoF trajectory estimation under IMU degradation.

    python main.py <imu0/data.csv | folder containing imu0/data.csv>
                   [-o estimated_trajectory.txt] [--blind 5.0]
                   [--metadata metadata.json]

Input : EuRoC IMU csv  (#timestamp [ns], wx, wy, wz, ax, ay, az)
Output: TUM pose file  (t[s] tx ty tz qx qy qz qw), one pose per input IMU row,
        followed by `--blind` seconds of dead-reckoned poses at the nominal IMU
        period (the "blind 5 s" after sensor death).

Design goals: single thread, streaming (O(1) memory), stdlib only (math, sys, os,
time, collections, itertools), pure-Python scalar maths (no numpy call overhead).

Pipeline
  1. Static alignment : mean gyro -> initial gyro bias; mean accel -> gravity
                        direction (roll/pitch), yaw = 0, |g| measured.
  2. Attitude         : quaternion integration (exact exp-map), Mahony-style
                        gravity correction gated on |a|~g and weighted down as the
                        measured accel noise grows; online roll/pitch gyro-bias
                        estimate (clamped).
  3. Position         : rotate accel to world, subtract g, integrate twice.
                        Zero-velocity updates (ZUPT) when gyro/accel are quiet.
  4. Dropouts         : gaps (dt > 2.5 x nominal) and invalid rows (NaN, garbage,
                        short lines) are bridged by damped constant-rate / constant-
                        velocity coasting, never by integrating a huge dt of data.
  5. Blind prediction : after the last sample, velocity (blend of filter velocity and
                        a least-squares slope of the last 1 s of positions) and
                        angular rate (EMA) are propagated with exponential damping.
"""
import sys
import os
import math
import time
from itertools import chain
from collections import deque

# ----------------------------- tunables ------------------------------------
G_NOM = 9.80665
ALIGN_SEC = 0.5          # static alignment window
ALIGN_MAX_RATE = 0.15    # rad/s  max |mean gyro| for the window to count as static
ALIGN_MAX_VAR = 0.05 ** 2  # (rad/s)^2 max total gyro variance in that window
W_MAX = 35.0             # rad/s   reject rows beyond this (garbage / spikes)
A_MAX = 160.0            # m/s^2
GAP_FACTOR = 2.5         # dt > GAP_FACTOR * nominal  -> dropout
KP0 = 4.0                # gravity feedback gain [1/s] (also soaks up horizontal accel bias)
KI = 0.05                # gyro-bias feedback gain
BIAS_CLAMP = 0.08        # rad/s max deviation of online gyro bias from initial
ZUPT_W2 = 0.04 ** 2      # (rad/s)^2   EMA of bias-corrected gyro power
ZUPT_DEV = 0.10          # m/s^2       EMA of | |a| - g |
ZUPT_DA2 = 0.20 ** 2     # (m/s^2)^2   EMA of |a_k - a_{k-1}|^2
ZUPT_HOLD_SEC = 0.30     # must be quiet this long before ZUPT fires
ZUPT_AW2 = 0.15 ** 2     # (m/s^2)^2 EMA of world-frame linear acceleration power
BIAS_REFRESH_W2 = 0.012 ** 2  # (rad/s)^2  gyro power below which the bias may be refreshed
BIAS_REFRESH_HOLD_SEC = 1.0   # ... and only after being still this long
LEAK_RATE_MAX = 1.0 / 5.0  # 1/s  velocity leak at the reference noise level (counters accel-bias drift)
LEAK_FLOOR = 1.0 / 60.0    # 1/s  constant minimum leak
LEAK_NOISE_REF = 0.25      # m/s^2 estimated accel noise sigma at which the leak reaches LEAK_RATE_MAX
TAU_V_GAP = 4.0          # s  velocity decay while coasting through a dropout
TAU_W_GAP = 2.0          # s  angular-rate decay while coasting
TAU_V_BLIND = 8.0        # s  velocity decay during the blind 5 s
TAU_W_BLIND = 2.0        # s  angular-rate decay during the blind 5 s
BLIND_VEL_WINDOW = 1.0   # s  window used for the blind-velocity regression
BLIND_BLEND = 0.5        # weight of the regression slope vs filter velocity
OMEGA_EMA_TAU = 0.3      # s


def resolve_input(path):
    if os.path.isdir(path):
        for cand in (os.path.join(path, "imu0", "data.csv"),
                     os.path.join(path, "data.csv")):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError("no imu0/data.csv under %s" % path)
    return path


def read_samples(path):
    """Yield (t_ns, (wx,wy,wz,ax,ay,az) | None). None = row exists but is unusable."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            p = line.split(",")
            try:
                t = int(p[0])
            except ValueError:
                try:
                    t = int(float(p[0]))
                except ValueError:
                    continue
            if len(p) < 7:
                yield t, None
                continue
            try:
                wx = float(p[1]); wy = float(p[2]); wz = float(p[3])
                ax = float(p[4]); ay = float(p[5]); az = float(p[6])
            except ValueError:
                yield t, None
                continue
            # NaN fails every comparison -> rejected here as well
            if (abs(wx) < W_MAX and abs(wy) < W_MAX and abs(wz) < W_MAX and
                    abs(ax) < A_MAX and abs(ay) < A_MAX and abs(az) < A_MAX):
                yield t, (wx, wy, wz, ax, ay, az)
            else:
                yield t, None


def rotq(qw, qx, qy, qz, rx, ry, rz):
    """q <- q (x) exp(r), r = rotation vector expressed in the body frame."""
    th2 = rx * rx + ry * ry + rz * rz
    if th2 < 1e-12:
        dw = 1.0 - th2 * 0.125
        s = 0.5 - th2 * (1.0 / 48.0)
    else:
        th = math.sqrt(th2)
        dw = math.cos(0.5 * th)
        s = math.sin(0.5 * th) / th
    dx = rx * s
    dy = ry * s
    dz = rz * s
    nw = qw * dw - qx * dx - qy * dy - qz * dz
    nx = qw * dx + qx * dw + qy * dz - qz * dy
    ny = qw * dy - qx * dz + qy * dw + qz * dx
    nz = qw * dz + qx * dy - qy * dx + qz * dw
    n = 1.0 / math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
    return nw * n, nx * n, ny * n, nz * n


def fmt_row(t_ns, px, py, pz, qw, qx, qy, qz):
    if qw < 0.0:
        qw = -qw; qx = -qx; qy = -qy; qz = -qz
    s, ns = divmod(t_ns, 1000000000)
    return "%d.%09d %.7f %.7f %.7f %.8f %.8f %.8f %.8f\n" % (
        s, ns, px, py, pz, qx, qy, qz, qw)


def run(in_path, out_path, blind_s=5.0):
    stream = read_samples(resolve_input(in_path))

    # ---------------- 1. static alignment ----------------
    buf = []
    n_ok = 0
    sw = [0.0, 0.0, 0.0]
    sa = [0.0, 0.0, 0.0]
    sw2 = 0.0
    t_first_ok = None
    for t, s in stream:
        buf.append((t, s))
        if s is None:
            continue
        if t_first_ok is None:
            t_first_ok = t
        n_ok += 1
        sw[0] += s[0]; sw[1] += s[1]; sw[2] += s[2]
        sw2 += s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
        sa[0] += s[3]; sa[1] += s[4]; sa[2] += s[5]
        if (t - t_first_ok) * 1e-9 >= ALIGN_SEC:
            break
        if len(buf) > 5000:
            break
    if n_ok == 0:
        raise RuntimeError("no valid IMU rows found")

    bg0x = sw[0] / n_ok; bg0y = sw[1] / n_ok; bg0z = sw[2] / n_ok
    # trust the window as "static" only if the gyro is quiet; otherwise start with
    # zero bias and let the online estimator / ZUPT find it
    m2 = bg0x * bg0x + bg0y * bg0y + bg0z * bg0z
    if m2 > ALIGN_MAX_RATE ** 2 or (sw2 / n_ok - m2) > ALIGN_MAX_VAR:
        bg0x = bg0y = bg0z = 0.0
    mx = sa[0] / n_ok; my = sa[1] / n_ok; mz = sa[2] / n_ok
    g = math.sqrt(mx * mx + my * my + mz * mz)
    if not (9.0 < g < 10.6):
        g = G_NOM

    roll = math.atan2(my, mz)
    pitch = math.atan2(-mx, math.sqrt(my * my + mz * mz))
    cr = math.cos(0.5 * roll); sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch); sp = math.sin(0.5 * pitch)
    qw = cr * cp
    qx = sr * cp
    qy = cr * sp
    qz = -sr * sp

    # nominal period from the first rows
    d = []
    for i in range(1, min(len(buf), 400)):
        x = buf[i][0] - buf[i - 1][0]
        if x > 0:
            d.append(x)
    if d:
        d.sort()
        nom_ns = d[len(d) // 2]
    else:
        nom_ns = 5000000
    nom = nom_ns * 1e-9
    gap_thr = GAP_FACTOR * nom

    # ---------------- state ----------------
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    bgx, bgy, bgz = bg0x, bg0y, bg0z
    omx = omy = omz = 0.0           # EMA of bias-corrected angular rate
    dev_ema = 0.0
    w2_ema = 0.0
    da2_ema = 0.0
    aw2_ema = 0.0
    pax = pay = paz = None
    static_cnt = 0
    static_need = max(3, int(ZUPT_HOLD_SEC / nom))
    refresh_need = max(10, int(BIAS_REFRESH_HOLD_SEC / nom))
    prev_t = None
    last_t = None
    hist = deque()                   # (t_s, px, py, pz) of the last BLIND_VEL_WINDOW s
    inv_leak_ref = 1.0 / LEAK_NOISE_REF
    exp = math.exp
    sqrt = math.sqrt

    out = open(out_path, "w")
    out.write("# timestamp tx ty tz qx qy qz qw\n")
    rows = []

    def coast(cdt):
        """Bridge a dropout of cdt seconds without sensor data (damped)."""
        nonlocal qw, qx, qy, qz, px, py, pz, vx, vy, vz
        if cdt <= 0.0:
            return
        k = TAU_W_GAP * (1.0 - exp(-cdt / TAU_W_GAP))
        qw, qx, qy, qz = rotq(qw, qx, qy, qz, omx * k, omy * k, omz * k)
        kv = TAU_V_GAP * (1.0 - exp(-cdt / TAU_V_GAP))
        px += vx * kv; py += vy * kv; pz += vz * kv
        dv = exp(-cdt / TAU_V_GAP)
        vx *= dv; vy *= dv; vz *= dv

    t0_ns = None
    for t, s in chain(buf, stream):
        if prev_t is None:
            t0_ns = t
            prev_t = t
            last_t = t
            hist.append((0.0, px, py, pz))
            rows.append(fmt_row(t, px, py, pz, qw, qx, qy, qz))
            continue
        dt_ns = t - prev_t
        if dt_ns <= 0:                      # duplicate / out-of-order row: skip
            continue
        dt = dt_ns * 1e-9
        prev_t = t
        last_t = t

        if s is None:                       # unusable row -> pure coasting
            coast(dt)
            static_cnt = 0
            pax = None
        else:
            if dt > gap_thr:                # dropout -> coast, then one nominal step
                coast(dt - nom)
                dt = nom
                static_cnt = 0
                pax = None
            ax = s[3]; ay = s[4]; az = s[5]
            wx = s[0] - bgx; wy = s[1] - bgy; wz = s[2] - bgz

            # --- noise / quietness trackers ---
            an = sqrt(ax * ax + ay * ay + az * az)
            dev = abs(an - g)
            dev_ema += 0.02 * (dev - dev_ema)
            w2_ema += 0.05 * ((wx * wx + wy * wy + wz * wz) - w2_ema)
            if pax is not None:
                ddx = ax - pax; ddy = ay - pay; ddz = az - paz
                da2_ema += 0.05 * ((ddx * ddx + ddy * ddy + ddz * ddz) - da2_ema)
            pax = ax; pay = ay; paz = az

            # --- gravity feedback (gated, weighted by measured noise) ---
            if an > 1e-3 and dev < 0.5 + 2.0 * dev_ema:
                inv = 1.0 / an
                axn = ax * inv; ayn = ay * inv; azn = az * inv
                vx_ = 2.0 * (qx * qz - qw * qy)
                vy_ = 2.0 * (qy * qz + qw * qx)
                vz_ = 1.0 - 2.0 * (qx * qx + qy * qy)
                ex = ayn * vz_ - azn * vy_
                ey = azn * vx_ - axn * vz_
                ez = axn * vy_ - ayn * vx_
                r = dev_ema * 2.0
                kp = KP0 / (1.0 + r * r)
                wx += kp * ex; wy += kp * ey; wz += kp * ez
                ki = KI * dt / (1.0 + r * r)
                bgx -= ki * ex; bgy -= ki * ey; bgz -= ki * ez
                if bgx > bg0x + BIAS_CLAMP: bgx = bg0x + BIAS_CLAMP
                elif bgx < bg0x - BIAS_CLAMP: bgx = bg0x - BIAS_CLAMP
                if bgy > bg0y + BIAS_CLAMP: bgy = bg0y + BIAS_CLAMP
                elif bgy < bg0y - BIAS_CLAMP: bgy = bg0y - BIAS_CLAMP
                if bgz > bg0z + BIAS_CLAMP: bgz = bg0z + BIAS_CLAMP
                elif bgz < bg0z - BIAS_CLAMP: bgz = bg0z - BIAS_CLAMP

            # smoothed rate for coasting / blind phase
            al = dt / OMEGA_EMA_TAU
            if al > 1.0:
                al = 1.0
            omx += al * (wx - omx); omy += al * (wy - omy); omz += al * (wz - omz)

            # --- attitude ---
            qw, qx, qy, qz = rotq(qw, qx, qy, qz, wx * dt, wy * dt, wz * dt)

            # --- position / velocity ---
            xx = qx * qx; yy = qy * qy; zz = qz * qz
            xy = qx * qy; xz = qx * qz; yz = qy * qz
            wxq = qw * qx; wyq = qw * qy; wzq = qw * qz
            awx = (1.0 - 2.0 * (yy + zz)) * ax + 2.0 * (xy - wzq) * ay + 2.0 * (xz + wyq) * az
            awy = 2.0 * (xy + wzq) * ax + (1.0 - 2.0 * (xx + zz)) * ay + 2.0 * (yz - wxq) * az
            awz = 2.0 * (xz - wyq) * ax + 2.0 * (yz + wxq) * ay + (1.0 - 2.0 * (xx + yy)) * az - g
            hdt2 = 0.5 * dt * dt
            px += vx * dt + awx * hdt2
            py += vy * dt + awy * hdt2
            pz += vz * dt + awz * hdt2
            vx += awx * dt; vy += awy * dt; vz += awz * dt
            # noise-adaptive weak velocity leak: ~0 on clean data, grows with degradation
            rr = sqrt(0.5 * da2_ema) * inv_leak_ref
            lk = 1.0 - dt * (LEAK_FLOOR + LEAK_RATE_MAX * (rr * rr if rr < 1.5 else 2.25))
            vx *= lk; vy *= lk; vz *= lk

            # --- zero-velocity update + (strict) bias refresh when quiet ---
            aw2_ema += 0.05 * ((awx * awx + awy * awy + awz * awz) - aw2_ema)
            if (w2_ema < ZUPT_W2 and dev_ema < ZUPT_DEV and da2_ema < ZUPT_DA2
                    and aw2_ema < ZUPT_AW2):
                static_cnt += 1
                if static_cnt >= static_need:
                    vx = vy = vz = 0.0
                    if static_cnt >= refresh_need and w2_ema < BIAS_REFRESH_W2:
                        bgx += 0.01 * (s[0] - bgx)
                        bgy += 0.01 * (s[1] - bgy)
                        bgz += 0.01 * (s[2] - bgz)
                        if abs(bgx - bg0x) > 2 * BIAS_CLAMP: bgx = bg0x
                        if abs(bgy - bg0y) > 2 * BIAS_CLAMP: bgy = bg0y
                        if abs(bgz - bg0z) > 2 * BIAS_CLAMP: bgz = bg0z
            else:
                static_cnt = 0

        ts = (t - t0_ns) * 1e-9
        hist.append((ts, px, py, pz))
        lim = ts - BLIND_VEL_WINDOW
        while hist[0][0] < lim:
            hist.popleft()
        rows.append(fmt_row(t, px, py, pz, qw, qx, qy, qz))
        if len(rows) >= 2000:
            out.write("".join(rows))
            rows = []

    # ---------------- 5. blind dead-reckoning ----------------
    n_blind = 0
    if blind_s > 0 and last_t is not None:
        # least-squares velocity over the last second of positions
        n = len(hist)
        bvx, bvy, bvz = vx, vy, vz
        if n >= 10 and hist[-1][0] - hist[0][0] > 0.2:
            tm = sum(h[0] for h in hist) / n
            sxx = sxp = syp = szp = 0.0
            mpx = sum(h[1] for h in hist) / n
            mpy = sum(h[2] for h in hist) / n
            mpz = sum(h[3] for h in hist) / n
            for h in hist:
                dtt = h[0] - tm
                sxx += dtt * dtt
                sxp += dtt * (h[1] - mpx)
                syp += dtt * (h[2] - mpy)
                szp += dtt * (h[3] - mpz)
            if sxx > 1e-9:
                b = BLIND_BLEND
                bvx = b * (sxp / sxx) + (1 - b) * vx
                bvy = b * (syp / sxx) + (1 - b) * vy
                bvz = b * (szp / sxx) + (1 - b) * vz
        vx, vy, vz = bvx, bvy, bvz
        wdec = exp(-nom / TAU_W_BLIND)
        vdec = exp(-nom / TAU_V_BLIND)
        n_blind = int(round(blind_s / nom))
        for k in range(1, n_blind + 1):
            omx *= wdec; omy *= wdec; omz *= wdec
            qw, qx, qy, qz = rotq(qw, qx, qy, qz, omx * nom, omy * nom, omz * nom)
            vx *= vdec; vy *= vdec; vz *= vdec
            px += vx * nom; py += vy * nom; pz += vz * nom
            rows.append(fmt_row(last_t + k * nom_ns, px, py, pz, qw, qx, qy, qz))
            if len(rows) >= 2000:
                out.write("".join(rows))
                rows = []

    if rows:
        out.write("".join(rows))
    out.close()
    return n_blind


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    in_path = argv[1]
    out_path = "estimated_trajectory.txt"
    blind = 5.0
    meta = None
    i = 2
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--out") and i + 1 < len(argv):
            out_path = argv[i + 1]; i += 2
        elif a == "--blind" and i + 1 < len(argv):
            blind = float(argv[i + 1]); i += 2
        elif a == "--metadata" and i + 1 < len(argv):
            meta = argv[i + 1]; i += 2
        else:
            print("unknown argument: %s" % a, file=sys.stderr)
            return 2

    w0 = time.perf_counter()
    c0 = time.process_time()
    try:
        nb = run(in_path, out_path, blind)
    except (OSError, RuntimeError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0

    n_rows = 0
    with open(out_path, "r") as f:
        for ln in f:
            if ln[0] != "#":
                n_rows += 1

    if meta:
        peak_kb = None
        try:
            import resource
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            pass
        n_in = n_rows - nb
        with open(meta, "w") as f:
            f.write("{\n")
            f.write('  "input_rows": %d,\n' % n_in)
            f.write('  "blind_rows": %d,\n' % nb)
            f.write('  "wall_seconds": %.4f,\n' % wall)
            f.write('  "cpu_seconds": %.4f,\n' % cpu)
            f.write('  "us_per_imu_sample": %.2f,\n' % (cpu * 1e6 / max(n_in, 1)))
            f.write('  "peak_rss_kb_incl_interpreter": %s,\n' %
                    ("null" if peak_kb is None else str(peak_kb)))
            f.write('  "python": "%s",\n' % sys.version.split()[0])
            f.write('  "threads": 1\n')
            f.write("}\n")
    print("wrote %d poses (%d blind) -> %s  [%.2fs]" % (n_rows, nb, out_path, wall))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
