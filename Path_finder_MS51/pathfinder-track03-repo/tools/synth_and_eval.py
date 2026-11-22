#!/usr/bin/env python3
"""
Local test harness (needs numpy; NOT part of the submitted runtime).

  python synth_and_eval.py gen  <outdir> [seed] [level]   # make imu0/data.csv + gt.txt
  python synth_and_eval.py eval <gt.txt> <estimated_trajectory.txt> [fail_t]

`gen` builds a smooth 3-D flight, simulates a 200 Hz IMU in EuRoC format and applies
a Track-03-style degradation: noise ramp, bias drift, a 2 s dropout and sensor
death. The ground truth keeps going for 5 s after death (the blind window).
`eval` aligns (Umeyama SE(3), like `evo -a`) and prints ATE RMSE plus short-window
RPE for: all poses, pre-failure poses, the dropout window, and the blind 5 s.
"""
import sys
import os
import numpy as np

HZ = 200.0
T_END = 45.0          # ground-truth length
T_FAIL = 40.0         # IMU dies here  -> 5 s blind
DROP = (22.0, 24.0)   # 2 s dropout
G = 9.80665


def euler_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(r), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp;     R[:, 2, 1] = cp * sr;                 R[:, 2, 2] = cp * cr
    return R


def R_to_quat(R):
    q = np.empty((len(R), 4))  # x y z w
    for i, M in enumerate(R):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s; x = (M[2, 1] - M[1, 2]) / s; y = (M[0, 2] - M[2, 0]) / s; z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            w = (M[2, 1] - M[1, 2]) / s; x = 0.25 * s; y = (M[0, 1] + M[1, 0]) / s; z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            w = (M[0, 2] - M[2, 0]) / s; x = (M[0, 1] + M[1, 0]) / s; y = 0.25 * s; z = (M[1, 2] + M[2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            w = (M[1, 0] - M[0, 1]) / s; x = (M[0, 2] + M[2, 0]) / s; y = (M[1, 2] + M[2, 1]) / s; z = 0.25 * s
        q[i] = (x, y, z, w) if w >= 0 else (-x, -y, -z, -w)
    return q


def gen(outdir, seed=0, level=1.0):
    rng = np.random.default_rng(seed)
    t0_ns = 1403636579758555392
    t = np.arange(0.0, T_END, 1.0 / HZ)
    h = 1e-4
    def env(tt):                       # static for 1 s, smooth ramp-in over 3 s (like EuRoC starts)
        u = np.clip((tt - 1.0) / 3.0, 0, 1)
        return u * u * (3 - 2 * u)
    def pos(tt):
        e = env(tt)
        return np.stack([4.0 * e * np.sin(0.31 * tt), 3.0 * e * np.sin(0.23 * tt + 0.7),
                         1.5 + 0.6 * e * np.sin(0.41 * tt + 1.1)], 1)
    def att(tt):
        e = env(tt)
        return euler_to_R(0.15 * e * np.sin(0.5 * tt), 0.10 * e * np.sin(0.35 * tt + 1.0), 0.8 * e * np.sin(0.1 * tt))
    P = pos(t)
    acc_w = (pos(t + h) - 2 * P + pos(t - h)) / h ** 2
    R = att(t)
    Rd = (att(t + h) - att(t - h)) / (2 * h)
    W = np.einsum("nji,njk->nik", R, Rd)                    # R^T Rdot (skew)
    gyro = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], 1)
    spec = np.einsum("nji,nj->ni", R, acc_w + np.array([0, 0, G]))  # specific force in body
    q = R_to_quat(R)

    # --- EuRoC-like baseline (always on) + progressive degradation (scaled by `level`) ---
    ramp = np.clip((t - 5.0) / (T_FAIL - 5.0), 0, 1)
    sg = 0.003 + 0.02 * ramp * level
    sa = 0.025 + 0.30 * ramp * level
    bg_const = np.array([-0.0022, 0.0208, 0.0757])       # roughly EuRoC MH-01 raw biases
    ba_const = np.array([-0.0236, 0.1210, 0.0781])
    bg = bg_const + level * (np.cumsum(rng.normal(0, 2e-4, (len(t), 3)), 0) * 0.5
                             + ramp[:, None] * np.array([0.01, -0.008, 0.012]))
    ba = ba_const + level * ramp[:, None] * np.array([0.08, -0.06, 0.05])
    gyro_m = gyro + bg + rng.normal(0, 1, gyro.shape) * sg[:, None]
    acc_m = spec + ba + rng.normal(0, 1, spec.shape) * sa[:, None]

    keep = (t < T_FAIL) & ~((t >= DROP[0]) & (t < DROP[1]))
    ns = (t0_ns + np.round(t * 1e9).astype(np.int64))
    os.makedirs(os.path.join(outdir, "imu0"), exist_ok=True)
    with open(os.path.join(outdir, "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for i in np.nonzero(keep)[0]:
            f.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (ns[i], *gyro_m[i], *acc_m[i]))
    with open(os.path.join(outdir, "gt.txt"), "w") as f:
        for i in range(len(t)):
            f.write("%d.%09d %.7f %.7f %.7f %.8f %.8f %.8f %.8f\n" %
                    (ns[i] // 10**9, ns[i] % 10**9, *P[i], *q[i]))
    print("rows in csv:", int(keep.sum()), " gt poses:", len(t))


def load(path):
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            rows.append([float(x) for x in ln.split()])
    return np.array(rows)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    return Rm, mu_d - Rm @ mu_s


def ate(gt, est, mask=None, align_on=None):
    # associate by nearest timestamp (est rows already at gt times)
    idx = np.searchsorted(gt[:, 0], est[:, 0])
    idx = np.clip(idx, 0, len(gt) - 1)
    ok = np.abs(gt[idx, 0] - est[:, 0]) < 2e-3
    g_, e_ = gt[idx[ok]], est[ok]
    Rm, tm = umeyama(e_[:, 1:4][align_on[ok]] if align_on is not None else e_[:, 1:4],
                     g_[:, 1:4][align_on[ok]] if align_on is not None else g_[:, 1:4])
    ea = (Rm @ e_[:, 1:4].T).T + tm
    err = np.linalg.norm(ea - g_[:, 1:4], axis=1)
    m = np.ones(len(err), bool) if mask is None else mask[ok]
    return float(np.sqrt(np.mean(err[m] ** 2))), float(np.max(err[m])), ok.sum()


def rpe(gt, est, delta_s=1.0, mask=None):
    idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 0, len(gt) - 1)
    ok = np.abs(gt[idx, 0] - est[:, 0]) < 2e-3
    g_, e_ = gt[idx[ok]], est[ok]
    k = int(delta_s * HZ)
    dg = g_[k:, 1:4] - g_[:-k, 1:4]
    de = e_[k:, 1:4] - e_[:-k, 1:4]
    # yaw/roll/pitch frame difference is removed by norm comparison of relative translations
    err = np.abs(np.linalg.norm(dg, axis=1) - np.linalg.norm(de, axis=1))
    m = np.ones(len(err), bool) if mask is None else mask[ok][k:]
    return float(np.sqrt(np.mean(err[m] ** 2)))


def evaluate(gt_path, est_path):
    gt, est = load(gt_path), load(est_path)
    t0 = gt[0, 0]
    rel = est[:, 0] - t0
    n_in = int((rel < T_FAIL - 1e-6).sum())
    print("estimated rows: %d  (pre-failure rows %d, blind rows %d)" % (len(est), n_in, len(est) - n_in))
    pre = rel < T_FAIL
    drop = (rel >= DROP[0]) & (rel < DROP[1] + 1.0)
    blind = rel >= T_FAIL
    for name, m in (("all poses", None), ("pre-failure", pre), ("post-dropout 1 s", drop), ("blind 5 s", blind)):
        a, mx, n = ate(gt, est, m, align_on=pre)
        print("ATE RMSE %-18s %7.3f m   (max %7.3f m, n=%d)" % (name, a, mx, n))
    print("RPE(1 s) translation, all poses: %.3f m" % rpe(gt, est, 1.0))
    # also report the number of poses per second sanity
    print("quat norm range: %.6f .. %.6f" % (np.linalg.norm(est[:, 4:8], axis=1).min(),
                                              np.linalg.norm(est[:, 4:8], axis=1).max()))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
    else:
        evaluate(sys.argv[2], sys.argv[3])
