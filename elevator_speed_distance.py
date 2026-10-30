#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电梯加速度数据处理：由 travel(加速度) + utc_time + air_pressure_pa 计算每秒速度与单趟位移

输入（只使用这 3 列，不使用任何供应商结果）：
  travel          : 每秒 100 个加速度采样，单位 cm/s²
  utc_time        : 时间戳（输出对齐用）
  air_pressure_pa : 气压，单位 Pa（仅用于判断运动段起止与方向）

输出（追加在原 CSV 上，原文件不改动）：
  speed_integral_cm_s / distance_integral_cm : 纯积分法（欧拉）
  speed_kf_cm_s       / distance_kf_cm       : 卡尔曼滤波 + 零速修正

约定：
  - 速度取绝对值（全部为正），单位 cm/s，四舍五入取整
  - 位移：上行为正、下行为负，单位 cm，四舍五入取整
  - 位移是"单趟位移"：一趟内从 0 累积到该趟总值，趟与趟之间（静止期）为 0
  - 每趟起点速度从 0 开始（v0 = 0）

算法要点：
  1) 气压 ±3 秒线性拟合斜率判断静止/运动；环境气压漂移约 0.02 Pa/s，趟内信号约 15 Pa/s，可分
  2) 静止块之间的区间即一趟运动；起点用加速度活跃度细化
  3) 纯积分：v += a·dt，s += v·dt + ½a·dt²（dt = 0.01 s）
  4) 卡尔曼滤波：状态 [位移, 速度, 加速度]，加速度作为观测，运动结束时以 v=0 做零速修正
  5) 趟末以 |v| 收敛到 0 判定停车，并把速度钉到 0，抑制残余漂移
"""

import numpy as np
import pandas as pd
import ast
import sys

DT = 0.01            # 加速度采样间隔 (s)，每秒 100 个采样
PRESSURE_WIN = 3     # 气压拟合半窗口 (行=秒)
SLOPE_STATIC_THR = 1.0   # |斜率| < 该值 (Pa/s) 判静止
START_ACTIVE_THR = 0.5   # 行内 |平均加速度| > 该值 (cm/s²) 判该行开始运动
END_ACTIVE_THR = 1.5     # 趟末：行内 |平均加速度| >= 该值 (cm/s²) 视为仍在减速


# ---------------------------------------------------------------- 工具
def round_half_away(x):
    """四舍五入（0.5 远离 0 进位），返回整数"""
    if x is None or not np.isfinite(x):
        return 0
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def load_travel(df):
    """解析 travel 列 -> list[np.ndarray]"""
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
    """滑动窗口线性拟合斜率 (Pa/s)，用于区分静止/运动"""
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
    """
    用气压斜率找静止块，静止块之间即一趟运动
    返回 trips: list[(start_row, coarse_end_row)]
    """
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
        if e - s + 1 < 5:          # 太短，不是一趟运行
            continue
        # 起点细化：第一个真正有加速度的行，再往前留 1 行（静止行，速度仍为 0）
        k = s
        while k <= e and abs(a_mean[k]) < START_ACTIVE_THR:
            k += 1
        trips.append((max(0, k - 1), e))
    return trips, np.abs(slope) >= SLOPE_STATIC_THR


# ---------------------------------------------------------------- 纯积分
def integrate_euler(travel, s, e):
    """返回该趟每行末的速度 v、位移 s"""
    v, pos = 0.0, 0.0
    vs, ss = [], []
    for k in range(s, e + 1):
        for a in travel[k]:
            pos += v * DT + 0.5 * a * DT * DT      # s = v0·t + ½·a·t²
            v += a * DT                            # v = v0 + a·t
        vs.append(v)
        ss.append(pos)
    return np.array(vs), np.array(ss)


# ---------------------------------------------------------------- 卡尔曼滤波
def integrate_kf(travel, s, e, sigma_a=2.0, q_scale=1.0, sigma_zv=1.5):
    """
    状态 x = [位移, 速度, 加速度]^T
    观测 z = 加速度计读数；运动结束时补充 v=0 的零速修正
    """
    F = np.array([[1.0, DT, 0.5 * DT * DT],
                  [0.0, 1.0, DT],
                  [0.0, 0.0, 1.0]])
    H = np.array([[0.0, 0.0, 1.0]])      # 观测加速度
    Hv = np.array([[0.0, 1.0, 0.0]])     # 零速观测
    Q = np.diag([(0.5 * sigma_a * DT * DT) ** 2,
                 (sigma_a * DT) ** 2,
                 (0.2 * sigma_a) ** 2]) * q_scale
    R = np.array([[sigma_a ** 2]])
    Rv = np.array([[sigma_zv ** 2]])
    I = np.eye(3)

    x = np.zeros((3, 1))
    P = np.diag([1.0, 1.0, 10.0])
    vs, ss = [], []
    for k in range(s, e + 1):
        for a in travel[k]:
            # 预测
            x = F @ x
            P = F @ P @ F.T + Q
            # 用加速度观测更新
            y = np.array([[a]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P
        vs.append(float(x[1, 0]))
        ss.append(float(x[0, 0]))
    return np.array(vs), np.array(ss)


# ---------------------------------------------------------------- 趟末收敛
def find_end(vs, a_mean, s, thr=END_ACTIVE_THR):
    """
    趟末判定：最后一个仍然处于减速的行（行内平均加速度足够大）。
    再回退 1 行，避免把"已经停稳但仍有余速"的那一行算进去。
    """
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
    """把每趟结果写回整段序列（静止期一律 0）"""
    n = len(travel)
    speed = np.zeros(n)
    dist = np.zeros(n)
    for s, e in trips:
        if method == 'euler':
            vs, ss = integrate_euler(travel, s, e)
        else:
            vs, ss = integrate_kf(travel, s, e, **kw)
        end = find_end(vs, a_mean, s)
        speed[s:s + end + 1] = np.abs(vs[:end + 1])
        dist[s:s + end + 1] = ss[:end + 1]      # 带符号：上行 + / 下行 -
    return speed, dist


# ---------------------------------------------------------------- 校验（仅读取参考列）
def report(df, trips, slope, v_int, d_int, v_kf, d_kf):
    if 'speed_cm_s' not in df.columns:
        return
    soft = df['speed_cm_s'].values.astype(float)
    dtrip = df['distance_trip_cm'].values.astype(float) if 'distance_trip_cm' in df.columns else None
    mask = soft > 5

    def speed_stat(v, name):
        d = np.abs(v - soft)
        print(f"  {name}: 运动中中位 {np.median(d[mask]):.2f} cm/s | <5cm/s {100*(d[mask]<5).mean():.1f}% "
              f"| <10cm/s {100*(d[mask]<10).mean():.1f}% | 相关 {np.corrcoef(v[mask], soft[mask])[0,1]:.4f}")

    print("\n【校验 · 只用供应商列做参考，不参与计算】")
    speed_stat(v_int, '纯积分')
    speed_stat(v_kf, '卡尔曼')

    # 方向校验：气压趋势 vs 位移符号
    bad = 0
    for s, e in trips:
        mid = (s + e) // 2
        if abs(slope[mid]) < SLOPE_STATIC_THR:
            continue
        if np.sign(-slope[mid]) != np.sign(d_int[s:e + 1][np.nonzero(d_int[s:e + 1])[0][-1]] if np.any(d_int[s:e + 1]) else 0):
            bad += 1
    print(f"  方向与气压趋势不一致的趟数: {bad}/{len(trips)}")

    # 位移校验：供应商 distance_trip_cm 在下一趟给出上一趟的行程值，用它做参考
    if dtrip is not None:
        refs = []
        for k, (s, e) in enumerate(trips):
            nxt = trips[k + 1][0] if k + 1 < len(trips) else -1
            refs.append(abs(dtrip[nxt]) if nxt >= 0 else np.nan)
        refs = np.array(refs, dtype=float)
        ours = []
        for s, e in trips:
            seg = d_int[s:e + 1]
            ours.append(abs(seg[np.nonzero(seg)[0][-1]]) if np.any(seg) else 0.0)
        ours = np.array(ours)
        ok = ~np.isnan(refs)
        err = np.abs(ours[ok] - refs[ok])
        print(f"  位移(纯积分): 与参考比值中位 {np.median(ours[ok]/refs[ok]):.3f} | "
              f"误差 中位 {np.median(err):.0f} cm 平均 {err.mean():.0f} cm 最大 {err.max():.0f} cm")


# ---------------------------------------------------------------- 主流程
def main(path, out=None):
    df = pd.read_csv(path)
    for col in ('travel', 'air_pressure_pa'):
        if col not in df.columns:
            raise SystemExit(f"缺少必需列: {col}")

    travel = load_travel(df)
    press = df['air_pressure_pa'].values.astype(float)
    a_mean = np.array([r.mean() for r in travel])

    slope = pressure_slope(press)
    trips, moving = detect_trips(a_mean, slope)

    print(f"行数 {len(df)} | 检测到运动趟数 {len(trips)}")
    lens = [e - s + 1 for s, e in trips]
    print(f"趟时长: 中位 {np.median(lens):.0f} s, 范围 {min(lens)}~{max(lens)} s")

    # 纯积分
    v_int, d_int = fill_rows(travel, trips, a_mean, 'euler')
    # 卡尔曼滤波 + 零速修正
    v_kf, d_kf = fill_rows(travel, trips, a_mean, 'kf', sigma_a=2.0, q_scale=1.0)

    # 写回（四舍五入）
    df['speed_integral_cm_s'] = [round_half_away(x) for x in v_int]
    df['distance_integral_cm'] = [round_half_away(x) for x in d_int]
    df['speed_kf_cm_s'] = [round_half_away(x) for x in v_kf]
    df['distance_kf_cm'] = [round_half_away(x) for x in d_kf]

    report(df, trips, slope, v_int, d_int, v_kf, d_kf)

    out = out or (path.rsplit('.', 1)[0] + '_computed.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n已输出: {out}")

    # 抽样展示
    print("\n【抽样 · 第 1 趟】")
    s, e = trips[0]
    print(f"{'行':>5} {'气压Pa':>8} {'参考速度':>8} {'积分速度':>8} {'卡尔曼速度':>8} {'参考Δ位':>9} {'积分位移':>9} {'卡尔曼位移':>9}")
    for k in range(s, min(e + 2, len(df))):
        ref_v = df['speed_cm_s'].iloc[k] if 'speed_cm_s' in df.columns else -1
        ref_d = df['distance_live_cm'].iloc[k] if 'distance_live_cm' in df.columns else np.nan
        print(f"{k:>5} {press[k]:>8.0f} {ref_v:>8.0f} {df['speed_integral_cm_s'].iloc[k]:>8d} "
              f"{df['speed_kf_cm_s'].iloc[k]:>8d} {ref_d:>9.0f} "
              f"{df['distance_integral_cm'].iloc[k]:>9d} {df['distance_kf_cm'].iloc[k]:>9d}")
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 elevator_speed_distance.py <csv> [输出csv]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
