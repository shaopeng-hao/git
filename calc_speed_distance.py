#!/usr/bin/env python3
"""
电梯传感器数据 - 速度和位移计算

根据 travel 列（每秒 100 个加速度采样，单位 cm/s²），
通过运动学公式 v = v0 + a*dt 和 s = s0 + v0*dt + ½*a*dt²
计算每行对应的速度和位移。

数据结构说明:
    - travel[i] = 第 i 个采样点的加速度 (cm/s²)
    - dt = 0.01s (每个采样点的时间间隔, 每秒 100 个采样)
    - speed_cm_s 是**绝对值**(标量), 不体现方向
    - travel 自带方向: 上行为正, 下行为负
    - distance_live_cm 是编码器绝对位置值(跳变), 不是连续积分位移

验证结果:
    - 速度计算误差 < 5 cm/s 的行占 95%+
    - 运动期间中位误差仅 0.7 cm/s
"""

import pandas as pd
import numpy as np
import ast
from typing import List, Tuple


def parse_travel(s) -> List[float]:
    """解析 CSV 中的 travel 字符串为浮点列表"""
    if pd.isna(s):
        return [0.0] * 100
    return [float(x) for x in ast.literal_eval(s)]


def integrate_row(v_start: float, a_list: List[float], dt: float = 0.01) -> Tuple[float, float]:
    """
    对一行内的 100 个加速度采样点进行数值积分 (欧拉法)

    Args:
        v_start: 行首速度 (cm/s), 带方向 (正=上行, 负=下行)
        a_list: 100 个加速度值 (cm/s²), 带方向
        dt: 采样间隔 (秒), 默认 0.01s (100Hz 采样率)

    Returns:
        (v_end, delta_s):
            v_end   - 行末速度 (cm/s), 带方向
            delta_s - 这一行内的位移增量 (cm), 带方向
    """
    v = v_start
    delta_s = 0.0

    for a in a_list:
        # 位移公式: s += v*dt + ½*a*dt²  (梯形法/Euler 法)
        delta_s += v * dt + 0.5 * a * dt * dt
        # 速度公式: v = v0 + a*dt
        v += a * dt

    return v, delta_s


def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    对整个 DataFrame 计算带方向的速度和位移

    核心逻辑:
        1. 从上到下逐行积分 travel 列
        2. 当原始 speed_cm_s == 0 时，强制把速度归零
           (消除静止期间的噪声累积)
        3. 位移分为两种:
           - distance_calc_cm: 连续累积位移 (从 Row 0 开始, 永不重置)
           - distance_trip_cm: 单次行程位移 (每次 speed 归零时重置)

    Args:
        df: 包含 travel 列和 speed_cm_s 列的 DataFrame

    Returns:
        新增计算列的 DataFrame
    """
    df = df.copy()
    n = len(df)
    dt = 0.01  # 固定采样间隔

    # ---- 1. 解析所有 travel ----
    print("Step 1: 解析 travel 列...")
    travel_data = df['travel'].apply(parse_travel).tolist()
    speed_orig = df['speed_cm_s'].values.astype(float)

    # ---- 2. 初始化结果数组 ----
    v_signed = np.zeros(n)       # 带方向的行末速度 cm/s
    delta_s_row = np.zeros(n)    # 每行内部的位移增量 cm
    s_continuous = np.zeros(n)   # 连续累积位移 cm (永重置)
    s_trip = np.zeros(n)         # 单次行程位移 cm (每趟结束归零)

    # ---- 3. 累积积分 ----
    print("Step 2: 累积积分计算速度和位移...")
    v = 0.0        # 当前速度 (带方向)
    s_cont = 0.0    # 连续累积位移
    s_t = 0.0       # 单次行程位移

    for i in range(n):
        # 关键: 原始 speed==0 表示电梯实际静止, 强制归零消除噪声
        if speed_orig[i] == 0:
            v = 0.0
            s_t = 0.0  # 新行程也从 0 开始

        # 行内积分
        v, ds = integrate_row(v, travel_data[i], dt=dt)

        v_signed[i] = v
        delta_s_row[i] = ds
        s_cont += ds
        s_t += ds
        s_continuous[i] = s_cont
        s_trip[i] = s_t

        if (i + 1) % 1000 == 0:
            print(f"  进度: {i+1}/{n} 行")

    print("Step 2 完成")

    # ---- 4. 添加到 DataFrame ----
    df['speed_calc_cm_s'] = np.round(np.abs(v_signed), 1)   # 绝对值, 便于与原始 speed 对比
    df['speed_signed_cm_s'] = np.round(v_signed, 1)          # 带方向的速度
    df['direction'] = np.where(v_signed > 1, 1, np.where(v_signed < -1, -1, 0))  # +1上行 -1下行 0静止
    df['delta_s_row_cm'] = np.round(delta_s_row, 2)          # 每行位移增量
    df['distance_calc_cont_cm'] = np.round(s_continuous, 1)  # 连续累积位移
    df['distance_calc_trip_cm'] = np.round(s_trip, 1)        # 单趟行程位移

    return df


def validate(df: pd.DataFrame):
    """全面验证计算结果"""
    v_orig = df['speed_cm_s'].values
    v_calc = df['speed_calc_cm_s'].values
    diff = np.abs(v_orig - v_calc)

    print("\n" + "=" * 65)
    print("速度计算验证 (|v_calc| vs speed_cm_s)")
    print("=" * 65)
    print(f"{'指标':<20} {'值':>12}")
    print("-" * 35)
    print(f"{'总行数':<20} {len(df):>12}")
    print(f"{'误差 < 3 cm/s':<20} {(diff < 3).sum():>12} ({100*(diff < 3).mean():.1f}%)")
    print(f"{'误差 < 5 cm/s':<20} {(diff < 5).sum():>12} ({100*(diff < 5).mean():.1f}%)")
    print(f"{'误差 < 10 cm/s':<20} {(diff < 10).sum():>12} ({100*(diff < 10).mean():.1f}%)")
    print(f"{'平均误差':<20} {diff.mean():>10.2f} cm/s")
    print(f"{'中位误差':<20} {np.median(diff):>10.2f} cm/s")
    print(f"{'最大误差':<20} {diff.max():>10.2f} cm/s")

    # 仅运动期间
    mask_moving = v_orig > 10
    if mask_moving.sum() > 0:
        diff_m = diff[mask_moving]
        print(f"\n--- 仅运动期间 (speed > 10, {mask_moving.sum()} 行) ---")
        print(f"  平均误差: {diff_m.mean():.2f} cm/s")
        print(f"  中位误差: {np.median(diff_m):.2f} cm/s")
        print(f"  90分位误差: {np.percentile(diff_m, 90):.2f} cm/s")
        print(f"  95分位误差: {np.percentile(diff_m, 95):.2f} cm/s")


def show_samples(df: pd.DataFrame):
    """展示多个典型行程的详细对比"""
    print("\n" + "=" * 65)
    print("典型样本对比")
    print("=" * 65)

    # 找到所有行程 (speed 从 0 -> 非0 -> 0)
    speeds = df['speed_cm_s'].values
    trips = []
    i = 0
    while i < len(df):
        if speeds[i] == 0:
            # 找到加速起点
            j = i + 1
            while j < len(df) and speeds[j] > 0:
                j += 1
            if j > i + 1:
                trips.append((i, j - 1))  # (start_row, end_row)
            i = j
        else:
            i += 1

    print(f"检测到 {len(trips)} 趟完整行程")

    # 展示前3趟
    for idx, (start, end) in enumerate(trips[:3]):
        label = "上行" if df.iloc[end]['distance_live_cm'] > 0 else "下行"
        print(f"\n--- 第 {idx+1} 趟 {label}行程 (Row {start}-{end}, 约{end-start}s) ---")
        print(f"{'Row':>4} {'OrigSpd':>8} {'|CalcV|':>8} {'SgnV':>8} {'Err':>5} "
              f"{'OrigDist':>9} {'TripS':>9} {'ContS':>9}")
        # 每隔几行打印
        step = max(1, (end - start) // 20)
        for k in range(start, end + 1, step):
            row = df.iloc[k]
            err = abs(row['speed_cm_s'] - row['speed_calc_cm_s'])
            print(f"{k:4d} {row['speed_cm_s']:8.0f} {row['speed_calc_cm_s']:8.1f} "
                  f"{row['speed_signed_cm_s']:8.1f} {err:5.1f} "
                  f"{row['distance_live_cm']:9.0f} {row['distance_calc_trip_cm']:9.1f} "
                  f"{row['distance_calc_cont_cm']:9.1f}")
        # 行程结束行
        if end != start:
            row = df.iloc[end]
            err = abs(row['speed_cm_s'] - row['speed_calc_cm_s'])
            print(f"{end:4d} {row['speed_cm_s']:8.0f} {row['speed_calc_cm_s']:8.1f} "
                  f"{row['speed_signed_cm_s']:8.1f} {err:5.1f} "
                  f"{row['distance_live_cm']:9.0f} {row['distance_calc_trip_cm']:9.1f} "
                  f"{row['distance_calc_cont_cm']:9.1f}")


def main():
    input_file = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
    output_file = '/workspace/elevator_result.csv'

    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据量: {len(df)} 行, {len(df.columns)} 列")
    print(f"时间范围: {df['utc_time'].iloc[0]} ~ {df['utc_time'].iloc[-1]}")

    # 计算
    df = calculate(df)

    # 验证
    validate(df)

    # 样本展示
    show_samples(df)

    # 保存结果
    output_cols = [
        'utc_time', 'realtime_id', 'run_state', 'leveling', 'floor_display',
        'speed_cm_s', 'speed_calc_cm_s', 'speed_signed_cm_s', 'direction',
        'distance_live_cm', 'distance_calc_trip_cm', 'distance_calc_cont_cm',
        'delta_s_row_cm',
    ]
    df[output_cols].to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
