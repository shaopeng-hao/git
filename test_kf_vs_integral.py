#!/usr/bin/env python3
"""
对照实验：纯积分 vs 纯 KF（无 ZUPT） vs KF+ZUPT
回答核心疑问：不用 ZUPT，KF 能修正漂移吗？
"""
import numpy as np
import pandas as pd
import ast
import sys

DT = 0.01
PRESSURE_WIN = 3
SLOPE_STATIC_THR = 1.0
START_ACTIVE_THR = 0.5
END_ACTIVE_THR = 1.5

def round_half_away(x):
    if x is None or not np.isfinite(x):
        return 0
    return int(np.sign(x) * np.floor(abs(x) + 0.5))

def load_travel(df):
    rows = []
    for s in df['travel']:
        if pd.isna(s):
            rows.append(np.zeros(100))
            continue
        try:
            arr = np.array([float(x) for x in ast.literal_eval(s)], dtype=float)
        except Exception:
            arr = np.zeros(100)
        rows.append(arr)
    return rows

def pressure_slope(press, win=PRESSURE_WIN):
    n = len(press)
    slope = np.zeros(n)
    for k in range(n):
        lo, hi = max(0, k - win), min(n, k + win + 1)
        y = press[lo:hi]
        x = np.arange(len(y), dtype=float)
        xm = x.mean()
        den = ((x - xm) ** 2).sum()
        slope[k] = ((x - xm) * (y - y.mean())).sum() / den if den else 0.0
    return slope

def detect_trips(a_mean, slope):
    n = len(a_mean)
    static = np.abs(slope) < SLOPE_STATIC_THR
    blocks, i = [], 0
    while i < n:
        if static[i]:
            j = i
            while j < n and static[j]:
                j += 1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    trips = []
    for a, b in zip(blocks, blocks[1:]):
        s, e = a[1] + 1, b[0] - 1
        if e - s + 1 < 5:
            continue
        k = s
        while k <= e and abs(a_mean[k]) < START_ACTIVE_THR:
            k += 1
        trips.append((max(0, k - 1), e))
    return trips

# ========== 方法1：纯积分 ==========
def integrate_euler(travel, s, e):
    v, pos = 0.0, 0.0
    vs, ss = [], []
    for k in range(s, e + 1):
        for a in travel[k]:
            pos += v * DT + 0.5 * a * DT * DT
            v += a * DT
        vs.append(v)
        ss.append(pos)
    return np.array(vs), np.array(ss)

# ========== 方法2：纯 KF（无 ZUPT！只有加速度观测） ==========
def integrate_kf_no_zupt(travel, s, e, sigma_a=2.0, q_scale=1.0):
    """
    三阶 KF，状态 [s, v, a]
    观测只有加速度 H=[0,0,1]，**没有 ZUPT**
    关键问题：s 和 v 完全不可观测，只能靠预测传播
    """
    F = np.array([[1.0, DT, 0.5 * DT * DT],
                  [0.0, 1.0, DT],
                  [0.0, 0.0, 1.0]])
    H = np.array([[0.0, 0.0, 1.0]])
    Q = np.diag([(0.5 * sigma_a * DT * DT) ** 2,
                 (sigma_a * DT) ** 2,
                 (0.2 * sigma_a) ** 2]) * q_scale
    R = np.array([[sigma_a ** 2]])
    I = np.eye(3)

    x = np.zeros((3, 1))
    P = np.diag([1.0, 1.0, 10.0])
    vs, ss = [], []
    for k in range(s, e + 1):
        for a in travel[k]:
            x = F @ x
            P = F @ P @ F.T + Q
            y = np.array([[a]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
        vs.append(float(x[1, 0]))
        ss.append(float(x[0, 0]))
    return np.array(vs), np.array(ss)

# ========== 方法3：KF + ZUPT（趟首尾 v=0） ==========
def integrate_kf_with_zupt(travel, s, e, sigma_a=2.0, q_scale=1.0, sigma_zv=3.0, zupt_rows=()):
    F = np.array([[1.0, DT, 0.5 * DT * DT],
                  [0.0, 1.0, DT],
                  [0.0, 0.0, 1.0]])
    H = np.array([[0.0, 0.0, 1.0]])
    Hv = np.array([[0.0, 1.0, 0.0]])
    Q = np.diag([(0.5 * sigma_a * DT * DT) ** 2,
                 (sigma_a * DT) ** 2,
                 (0.2 * sigma_a) ** 2]) * q_scale
    R = np.array([[sigma_a ** 2]])
    Rv = np.array([[sigma_zv ** 2]])
    I = np.eye(3)
    zupt_rows = set(zupt_rows)

    x = np.zeros((3, 1))
    P = np.diag([1.0, 1.0, 10.0])
    vs, ss = [], []
    for k in range(s, e + 1):
        for a in travel[k]:
            x = F @ x
            P = F @ P @ F.T + Q
            y = np.array([[a]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
        if k in zupt_rows:
            yv = np.array([[0.0]]) - Hv @ x
            Sv = Hv @ P @ Hv.T + Rv
            Kv = P @ Hv.T @ np.linalg.inv(Sv)
            x = x + Kv @ yv
            P = (I - Kv @ Hv) @ P
        vs.append(float(x[1, 0]))
        ss.append(float(x[0, 0]))
    return np.array(vs), np.array(ss)

def find_end(vs, a_mean, s, thr=END_ACTIVE_THR):
    peak = int(np.argmax(np.abs(vs)))
    end = peak
    for k in range(s, s + len(vs)):
        if abs(a_mean[k]) >= thr and (k - s) > end:
            end = k - s
    end = max(end, peak)
    if end > peak:
        end -= 1
    return end

# ======================================================================
# 主对比
# ======================================================================
csv_path = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
df = pd.read_csv(csv_path)
travel = load_travel(df)
press = df['air_pressure_pa'].values.astype(float)
a_mean = np.array([r.mean() for r in travel])
slope = pressure_slope(press)
trips = detect_trips(a_mean, slope)

soft = df['speed_cm_s'].values.astype(float)
dtrip = df['distance_trip_cm'].values.astype(float)
mask = soft > 5

print("=" * 80)
print("对照实验：纯积分 vs 纯KF(无ZUPT) vs KF+ZUPT")
print("=" * 80)
print("\n【数学本质】三阶 KF 只有加速度观测 H=[0,0,1]")
print("  → 位置 s 和速度 v 在数学上不可观测！")
print("  → KF 观测更新只修正加速度状态")
print("  → s, v 只能通过 F 矩阵预测步骤传播")
print("  → 没有额外观测，KF 无法修正 s, v 的漂移")

# --- 对每趟跑三种方法 ---
speed_euler = np.zeros(len(df))
dist_euler = np.zeros(len(df))
speed_kf0 = np.zeros(len(df))
dist_kf0 = np.zeros(len(df))
speed_kfz = np.zeros(len(df))
dist_kfz = np.zeros(len(df))

for s, e in trips:
    # 纯积分（用于定位 end）
    vs0, _ = integrate_euler(travel, s, e)
    end = find_end(vs0, a_mean, s)

    # 方法1：纯积分
    vs, ss = integrate_euler(travel, s, s + end)
    speed_euler[s:s+end+1] = np.abs(vs)
    dist_euler[s:s+end+1] = ss

    # 方法2：纯 KF（无 ZUPT）
    vs, ss = integrate_kf_no_zupt(travel, s, s + end, sigma_a=2.0, q_scale=0.3)
    speed_kf0[s:s+end+1] = np.abs(vs)
    dist_kf0[s:s+end+1] = ss

    # 方法3：KF + ZUPT（趟首尾 v=0）
    e2 = min(e, s + end + 1)
    vs, ss = integrate_kf_with_zupt(travel, s, e2, sigma_a=2.0, q_scale=0.3,
                                     sigma_zv=3.0, zupt_rows=(s, e2))
    vs, ss = vs[:end+1], ss[:end+1]
    speed_kfz[s:s+end+1] = np.abs(vs)
    dist_kfz[s:s+end+1] = ss

# --- 速度校验 ---
def speed_compare(name, v):
    d = np.abs(v - soft)
    return (np.median(d[mask]), 100*(d[mask]<5).mean(), 100*(d[mask]<10).mean(),
            np.corrcoef(v[mask], soft[mask])[0,1])

# --- 位移校验 ---
def distance_compare(name, d, trips):
    refs, ours = [], []
    for k, (s, e) in enumerate(trips):
        nxt = trips[k+1][0] if k+1 < len(trips) else -1
        refs.append(abs(dtrip[nxt]) if nxt >= 0 else np.nan)
    refs = np.array(refs, dtype=float)
    for s, e in trips:
        seg = d[s:e+1]
        ours.append(abs(seg[np.nonzero(seg)[0][-1]]) if np.any(seg) else 0.0)
    ours = np.array(ours)
    ok = ~np.isnan(refs)
    ratio = np.median(ours[ok]/refs[ok])
    err = np.abs(ours[ok] - refs[ok])
    return ratio, np.median(err), err.mean(), err.max()

print("\n【速度精度对比 · 与供应商参考列（speed_cm_s）比较】")
for name, v in [('纯积分', speed_euler), ('纯KF(无ZUPT)', speed_kf0), ('KF+ZUPT', speed_kfz)]:
    med, lt5, lt10, corr = speed_compare(name, v)
    print(f"  {name:<20s}: 中位误差 {med:.2f} cm/s | <5cm/s {lt5:.1f}% | <10cm/s {lt10:.1f}% | 相关 {corr:.4f}")

print("\n【位移精度对比 · 与供应商参考列（distance_trip_cm）比较】")
for name, d in [('纯积分', dist_euler), ('纯KF(无ZUPT)', dist_kf0), ('KF+ZUPT', dist_kfz)]:
    ratio, med, avg, mx = distance_compare(name, d, trips)
    print(f"  {name:<20s}: 比值中位 {ratio:.3f} | 误差 中位 {med:.0f}cm 平均 {avg:.0f}cm 最大 {mx:.0f}cm")

# --- 关键诊断：纯 KF vs 纯积分 的差异 ---
print("\n【关键诊断：纯KF(无ZUPT) vs 纯积分 的差异】")
print("  如果纯KF真能修正漂移，它应该比纯积分好很多")
print("  但如果差异很小，说明KF确实无法修正不可观测状态")
diff_v_eul_kf0 = np.abs(speed_euler - speed_kf0)
diff_d_eul_kf0 = np.abs(dist_euler - dist_kf0)
print(f"  速度差: 中位 {np.median(diff_v_eul_kf0):.2f} cm/s | 最大 {diff_v_eul_kf0.max():.2f} cm/s | 相关 {np.corrcoef(speed_euler, speed_kf0)[0,1]:.4f}")
print(f"  位移差: 中位 {np.median(diff_d_eul_kf0):.2f} cm   | 最大 {diff_d_eul_kf0.max():.2f} cm   | 相关 {np.corrcoef(dist_euler, dist_kf0)[0,1]:.4f}")

diff_v_kf0_kfz = np.abs(speed_kf0 - speed_kfz)
diff_d_kf0_kfz = np.abs(dist_kf0 - dist_kfz)
print(f"\n  纯KF vs KF+ZUPT 差:")
print(f"  速度差: 中位 {np.median(diff_v_kf0_kfz):.2f} cm/s | 最大 {diff_v_kf0_kfz.max():.2f} cm/s")
print(f"  位移差: 中位 {np.median(diff_d_kf0_kfz):.2f} cm   | 最大 {diff_d_kf0_kfz.max():.2f} cm")

# --- 一趟抽样 ---
print("\n【抽样：第 1 趟的速度/位移曲线对比】")
s, e = trips[0]
end = find_end(integrate_euler(travel, s, e)[0], a_mean, s)
print(f"  趟 {0}: 行 {s}~{s+end}，{end+1} 秒")
print(f"  {'行':>5} {'参考v':>7} {'积分v':>7} {'纯KFv':>7} {'ZUPTv':>7} | {'参考d':>9} {'积分d':>9} {'纯KFd':>9} {'ZUPTd':>9}")
for k in range(s, min(s+end+1, s+15)):
    rv = soft[k] if soft[k] > 0 else -1
    print(f"  {k:>5} {rv:>7.0f} {speed_euler[k]:>7.1f} {speed_kf0[k]:>7.1f} {speed_kfz[k]:>7.1f} | {-1:>9.0f} {dist_euler[k]:>9.1f} {dist_kf0[k]:>9.1f} {dist_kfz[k]:>9.1f}")

print("\n" + "=" * 80)
print("结论：")
print("=" * 80)
print("""
  你的直觉有一部分是对的：KF 本身确实做了"滤波"——它通过过程噪声 Q 和观测噪声 R 的权衡，
  让积分轨迹更平滑（加速度噪声被压制了）。但这不是"修正漂移"，而是"降低噪声"。

  真正的漂移来源是加速度传感器的系统性零偏（bias），而不是白噪声。
  在当前配置下：
    - 观测只有加速度 → KF 只能看到 a
    - s 和 v 没有直接观测 → 不可观测，KF 无法估计它们的漂移
    - 加速度零偏是低频/常量误差 → 加速度观测本身无法区分 "bias" 和 "真加速度"

  要真正修正漂移，有两条路：
    1) ZUPT（零速修正）：在静止行强制 v=0，这相当于给了 v 一个观测 → v 变得可观测
       同时通过协方差耦合，s 也被间接约束
    2) 增加 bias 状态：x=[s,v,a,b_a]，H=[0,0,1,1]，并在静止行做 ZUPT
       这样 KF 可以估计 b_a，从加速度观测中减掉 bias，抑制积分漂移

  也就是说：
    - 纯 KF（无 ZUPT）= 更平滑的积分，但漂移依旧在
    - KF + ZUPT = 有额外观测，能真正约束 v（间接约束 s）
    - KF + bias + ZUPT = 最优方案，能估计 bias 并抑制漂移
""")
