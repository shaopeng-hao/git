#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电梯加速度数据处理 - 多 sheet 并行输出 + 准确率评估

功能：
  1) 读取原始 CSV（含 travel + air_pressure_pa）
  2) 运行纯积分、卡尔曼滤波（KF+ZUPT）两种算法
  3) 输出合并 CSV（所有趟结果在一张表）
  4) 输出分趟 CSV（每趟一个文件，放在 trips/ 子目录下）
  5) 输出准确率评估表 accuracy_assessment.csv

准确率容差（可在 ACCURACY_SPEED_TOL / ACCURACY_DIST_TOL 处调整）：
  速度: |计算速度 - 供应商速度| ≤ 5 cm/s
  位移: |计算位移 - 供应商位移| ≤ 20 cm  (逐行比)
"""

import numpy as np
import pandas as pd
import ast
import sys
import os
import time

# ========== 参数 ==========
DT = 0.01
PRESSURE_WIN = 3
SLOPE_STATIC_THR = 1.0
START_ACTIVE_THR = 0.5
END_ACTIVE_THR = 1.5

# 准确率容差
ACCURACY_SPEED_TOL = 5.0     # cm/s
ACCURACY_DIST_TOL  = 20.0    # cm (逐行位移)

# KF 参数
KF_SIGMA_A  = 2.0
KF_Q_SCALE  = 0.3
KF_SIGMA_ZV = 3.0


# ========== 工具函数 ==========
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
    return trips, np.abs(slope) >= SLOPE_STATIC_THR


# ========== 纯积分 ==========
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


# ========== 卡尔曼（带 ZUPT） ==========
def integrate_kf(travel, s, e, sigma_a=KF_SIGMA_A, q_scale=KF_Q_SCALE,
                 sigma_zv=KF_SIGMA_ZV, zupt_rows=()):
    F = np.array([[1.0, DT, 0.5 * DT * DT],
                  [0.0, 1.0, DT],
                  [0.0, 0.0, 1.0]])
    H  = np.array([[0.0, 0.0, 1.0]])
    Hv = np.array([[0.0, 1.0, 0.0]])
    Q = np.diag([(0.5 * sigma_a * DT * DT) ** 2,
                 (sigma_a * DT) ** 2,
                 (0.2 * sigma_a) ** 2]) * q_scale
    R  = np.array([[sigma_a ** 2]])
    Rv = np.array([[sigma_zv ** 2]])
    I  = np.eye(3)
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


def fill_rows(travel, trips, a_mean, method, **kw):
    n = len(travel)
    speed = np.zeros(n)
    dist  = np.zeros(n)
    for s, e in trips:
        if method == 'euler':
            vs, ss = integrate_euler(travel, s, e)
            end = find_end(vs, a_mean, s)
        else:
            vs0, _ = integrate_euler(travel, s, e)
            end = find_end(vs0, a_mean, s)
            e2 = min(e, s + end + 1)
            vs, ss = integrate_kf(travel, s, e2, zupt_rows=(s, e2), **kw)
            vs, ss = vs[:end + 1], ss[:end + 1]
        speed[s:s + end + 1] = np.abs(vs[:end + 1])
        dist[s:s + end + 1]  = ss[:end + 1]
    return speed, dist


# ========== 准确率计算 ==========
def calc_trip_accuracy(trip_df):
    """
    trip_df: 单个行程的 DataFrame，需含列:
      naive_speed_cm_s, kalman_speed_cm_s, supplier_speed_cm_s
      naive_displacement_cm, kalman_displacement_cm, supplier_displacement_cm

    准确率规则:
      速度: 逐行比 |计算速度 - 供应商速度| ≤ 5 cm/s
      位移: 只在供应商位移"发生更新"的行比较（跳过平台段）| |计算位移| - |供应商位移| | ≤ 20 cm
            + 额外报告趟末总位移误差
    """
    n = len(trip_df)
    if n == 0:
        return {
            '总行数': 0,
            '朴素积分速度准确率': '0.0%', '朴素积分速度在容差内行数': 0,
            '卡尔曼速度准确率':   '0.0%', '卡尔曼速度在容差内行数': 0,
            '朴素积分位移准确率': '0.0%', '朴素积分位移在容差内行数': 0,
            '卡尔曼位移准确率':   '0.0%', '卡尔曼位移在容差内行数': 0,
            '朴素积分位移更新点数': 0, '卡尔曼位移更新点数': 0,
            '趟末总位移_朴素': 0, '趟末总位移_KF': 0, '趟末总位移_参考': 0,
        }

    ref_v = trip_df['supplier_speed_cm_s'].values.astype(float)
    ref_d = np.abs(trip_df['supplier_displacement_cm'].values.astype(float))

    naive_v = trip_df['naive_speed_cm_s'].values.astype(float)
    naive_d = np.abs(trip_df['naive_displacement_cm'].values.astype(float))
    kf_v    = trip_df['kalman_speed_cm_s'].values.astype(float)
    kf_d    = np.abs(trip_df['kalman_displacement_cm'].values.astype(float))

    # 速度：逐行
    naive_v_ok = np.abs(naive_v - ref_v) <= ACCURACY_SPEED_TOL
    kf_v_ok    = np.abs(kf_v    - ref_v) <= ACCURACY_SPEED_TOL

    # 位移：只在供应商位移"变化了"的行做比较（跳过平台）
    d_changed = np.concatenate([[True], np.abs(np.diff(ref_d)) > 0.5])
    # 也包含最后一行
    d_changed[-1] = True

    naive_d_ok = (np.abs(naive_d - ref_d) <= ACCURACY_DIST_TOL) & d_changed
    kf_d_ok    = (np.abs(kf_d    - ref_d) <= ACCURACY_DIST_TOL) & d_changed
    n_d_update = int(d_changed.sum())   # 供应商位移更新点数量

    naive_v_n = int(naive_v_ok.sum())
    kf_v_n    = int(kf_v_ok.sum())
    naive_d_n = int(naive_d_ok.sum())
    kf_d_n    = int(kf_d_ok.sum())

    # 趟末总位移（绝对值）
    last_idx = -1
    naive_end = naive_d[last_idx]
    kf_end    = kf_d[last_idx]
    ref_end   = ref_d[last_idx]

    def pct(a, total=n):
        return f"{100.0 * a / total:.1f}%"

    def pct_d(a):
        return f"{100.0 * a / n_d_update:.1f}%" if n_d_update > 0 else "N/A"

    return {
        '总行数': n,
        '朴素积分速度准确率': pct(naive_v_n),
        '朴素积分速度在容差内行数': naive_v_n,
        '卡尔曼速度准确率':   pct(kf_v_n),
        '卡尔曼速度在容差内行数':   kf_v_n,
        '朴素积分位移准确率': pct_d(naive_d_n),
        '朴素积分位移在容差内行数': naive_d_n,
        '卡尔曼位移准确率':   pct_d(kf_d_n),
        '卡尔曼位移在容差内行数':   kf_d_n,
        '朴素积分位移更新点数': n_d_update,
        '趟末总位移_朴素cm': round(naive_end),
        '趟末总位移_KFcm':    round(kf_end),
        '趟末总位移_参考cm':  round(ref_end),
        '趟末总位移误差_朴素cm': round(abs(naive_end - ref_end)),
        '趟末总位移误差_KFcm':   round(abs(kf_end    - ref_end)),
    }


# ========== 主流程 ==========
def main(csv_path, out_dir=None):
    t0 = time.time()
    print(f"[1/4] 读取原始数据...", flush=True)
    df = pd.read_csv(csv_path)
    for col in ('travel', 'air_pressure_pa'):
        if col not in df.columns:
            raise SystemExit(f"缺少必需列: {col}")

    travel = load_travel(df)
    press  = df['air_pressure_pa'].values.astype(float)
    a_mean = np.array([r.mean() for r in travel])

    print(f"  行数 {len(df)} | 耗时 {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"[2/4] 趟检测 + 算法计算...", flush=True)
    slope = pressure_slope(press)
    trips, moving = detect_trips(a_mean, slope)
    print(f"  检测到 {len(trips)} 趟 | 耗时 {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    v_int, d_int = fill_rows(travel, trips, a_mean, 'euler')
    v_kf,  d_kf  = fill_rows(travel, trips, a_mean, 'kf',
                             sigma_a=KF_SIGMA_A, q_scale=KF_Q_SCALE, sigma_zv=KF_SIGMA_ZV)
    print(f"  纯积分 + KF 完成 | 耗时 {time.time() - t2:.1f}s", flush=True)

    # 取供应商参考列（用于准确率计算）
    supplier_v = df['speed_cm_s'].values.astype(float) if 'speed_cm_s' in df.columns else np.zeros(len(df))
    # distance_live_cm = 逐行累积位移（趟内），distance_trip_cm = 趟末总行程（常量）
    supplier_d = df['distance_live_cm'].values.astype(float) if 'distance_live_cm' in df.columns else np.zeros(len(df))

    print(f"[3/4] 写出结果...", flush=True)
    out_dir = out_dir or os.path.dirname(csv_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # --- 3a. 合并 CSV ---
    df['speed_integral_cm_s']      = [round_half_away(x) for x in v_int]
    df['distance_integral_cm']     = [round_half_away(x) for x in d_int]
    df['speed_kf_cm_s']            = [round_half_away(x) for x in v_kf]
    df['distance_kf_cm']           = [round_half_away(x) for x in d_kf]

    combined_path = os.path.join(out_dir, 'elevator_computed_all.csv')
    df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 合并 CSV: {combined_path}", flush=True)

    # --- 3b. 分趟 CSV（每行一个文件，放在 trips/ 子目录）---
    trips_dir = os.path.join(out_dir, 'trips')
    os.makedirs(trips_dir, exist_ok=True)

    # 准备连续绝对位移（趟末归零的单趟位移）
    # 注意：d_int / d_kf 已经是带符号的单趟位移，趟间静止期为 0

    all_trip_frames = []
    acc_rows = []

    for tidx, (s, e) in enumerate(trips):
        # 找到 end（趟末行）
        vs0, _ = integrate_euler(travel, s, e)
        end = find_end(vs0, a_mean, s)
        end_row = s + end

        # 构建单趟 DataFrame（只含运动段行）
        n_trip = end_row - s + 1
        trip_rows = range(s, end_row + 1)

        # 气压方向
        mid_slope = slope[(s + end_row) // 2]
        pressure_direction = 'UP' if mid_slope < -SLOPE_STATIC_THR else ('DOWN' if mid_slope > SLOPE_STATIC_THR else 'STATIC')

        motion_status = ['MOTION'] * n_trip
        motion_status[0]  = 'START'
        motion_status[-1] = 'END'

        trip_df = pd.DataFrame({
            'source_row':          list(trip_rows),
            'utc_time':            df['utc_time'].iloc[s:end_row+1].values if 'utc_time' in df.columns else [f"t_{i}" for i in range(n_trip)],
            'air_pressure_pa':     press[s:end_row+1].astype(int),
            'trip_id':             [tidx + 1] * n_trip,
            'motion_status':       motion_status,
            'pressure_direction':  [pressure_direction] * n_trip,
            # KF（卡尔曼）结果 —— 注意位移带方向符号
            'kalman_speed_cm_s':      [round_half_away(x) for x in v_kf[s:end_row+1]],
            'kalman_displacement_cm': [round_half_away(x) for x in d_kf[s:end_row+1]],
            # 朴素积分结果
            'naive_speed_cm_s':       [round_half_away(x) for x in v_int[s:end_row+1]],
            'naive_displacement_cm':  [round_half_away(x) for x in d_int[s:end_row+1]],
            # 供应商参考
            'supplier_speed_cm_s':    [round_half_away(x) for x in supplier_v[s:end_row+1]],
            'supplier_displacement_cm':[round_half_away(x) for x in supplier_d[s:end_row+1]],
        })

        # 写出单趟 CSV
        trip_csv = os.path.join(trips_dir, f'行程_{tidx + 1}.csv')
        trip_df.to_csv(trip_csv, index=False, encoding='utf-8-sig')

        # 收集准确性评估
        acc = calc_trip_accuracy(trip_df)
        acc['行程ID'] = tidx + 1
        acc_rows.append(acc)

    print(f"  ✓ {len(trips)} 个分趟 CSV: {trips_dir}/", flush=True)

    # --- 3c. 准确性评估 ---
    acc_df = pd.DataFrame(acc_rows)
    # 调整列顺序
    cols = ['行程ID', '总行数',
            '朴素积分速度准确率', '朴素积分速度在容差内行数',
            '卡尔曼速度准确率',   '卡尔曼速度在容差内行数',
            '朴素积分位移准确率', '朴素积分位移在容差内行数',
            '卡尔曼位移准确率',   '卡尔曼位移在容差内行数',
            '朴素积分位移更新点数',
            '趟末总位移_朴素cm', '趟末总位移_KFcm', '趟末总位移_参考cm',
            '趟末总位移误差_朴素cm', '趟末总位移误差_KFcm']
    acc_df = acc_df[cols]
    acc_path = os.path.join(out_dir, 'accuracy_assessment.csv')
    acc_df.to_csv(acc_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 准确率评估: {acc_path}", flush=True)

    # 汇总
    print(f"\n[4/4] 准确率汇总（速度容差 ≤ {ACCURACY_SPEED_TOL} cm/s，位移容差 ≤ {ACCURACY_DIST_TOL} cm）", flush=True)

    # 4a. 速度准确率（逐行）
    mask = supplier_v > 5
    naive_v_acc = 100.0 * (np.abs(v_int[mask] - supplier_v[mask]) <= ACCURACY_SPEED_TOL).mean()
    kf_v_acc    = 100.0 * (np.abs(v_kf[mask]  - supplier_v[mask]) <= ACCURACY_SPEED_TOL).mean()
    print(f"  速度准确率（逐行，supplier_speed > 5 的 {mask.sum()} 行）: "
          f"朴素积分 {naive_v_acc:.1f}% | 卡尔曼 {kf_v_acc:.1f}%", flush=True)

    # 4b. 趟末总位移误差（更公平的位移指标）
    naive_end_err = np.array([a['趟末总位移误差_朴素cm'] for a in acc_rows])
    kf_end_err    = np.array([a['趟末总位移误差_KFcm'] for a in acc_rows])
    print(f"  趟末总位移误差（{len(acc_rows)} 趟）: "
          f"朴素积分 中位 {np.median(naive_end_err):.0f}cm 平均 {naive_end_err.mean():.0f}cm 最大 {naive_end_err.max():.0f}cm | "
          f"卡尔曼 中位 {np.median(kf_end_err):.0f}cm 平均 {kf_end_err.mean():.0f}cm 最大 {kf_end_err.max():.0f}cm", flush=True)

    # 趟级位移准确率（只在供应商位移更新点）
    naive_d_trip = []; kf_d_trip = []
    for a in acc_rows:
        n_upd = int(a.get('朴素积分位移更新点数', 0))
        if n_upd > 0:
            naive_d_trip.append(int(a['朴素积分位移在容差内行数']) / n_upd)
            kf_d_trip.append(int(a['卡尔曼位移在容差内行数'])   / n_upd)
    if naive_d_trip:
        print(f"  位移准确率（趟级平均，仅供应商位移更新点）: "
              f"朴素积分 {100*np.mean(naive_d_trip):.1f}% | 卡尔曼 {100*np.mean(kf_d_trip):.1f}%", flush=True)

    # 趟级速度准确率
    naive_v_trip = []; kf_v_trip = []
    for a in acc_rows:
        n = int(a['总行数'])
        if n > 0:
            naive_v_trip.append(int(a['朴素积分速度在容差内行数']) / n)
            kf_v_trip.append(int(a['卡尔曼速度在容差内行数'])   / n)
    print(f"  速度准确率（趟级平均）: "
          f"朴素积分 {100*np.mean(naive_v_trip):.1f}% | 卡尔曼 {100*np.mean(kf_v_trip):.1f}%", flush=True)

    print(f"\n总耗时 {time.time() - t0:.1f}s", flush=True)
    return combined_path, trips_dir, acc_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 elevator_batch_run.py <原始CSV> [输出目录]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
