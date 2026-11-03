#!/usr/bin/env python3
"""
把 elevator_computed.csv 按段拆成多 sheet 的 Excel，按参考 Excel 的列布局，
并计算每一行（及每一行所在趟）的准确率。
"""
import sys, ast
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')
from elevator_speed_distance import (
    load_travel, pressure_slope, detect_trips, SLOPE_STATIC_THR
)

DT = 0.01
SPEED_TOL = 5.0       # 速度容差 cm/s（绝对误差 ≤ 5 视为准确）
DIST_TOL = 50.0       # 位移容差 cm（绝对误差 ≤ 50 视为准确）

# ========== 1. 读取 & 检测趟 ==========
src_path  = '.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
comp_path = '.uploads/759f5ff6-e3ac-408e-8e8d-3a00d147b71e_elevator_computed.csv'

src  = pd.read_csv(src_path)
comp = pd.read_csv(comp_path)
# 用原始数据的 utc_time + air_pressure_pa + travel，合并 computed 的 4 个计算列
df = pd.DataFrame({
    'utc_time':               src['utc_time'],
    'air_pressure_pa':        src['air_pressure_pa'],
    'travel':                 src['travel'],
    'speed_integral_cm_s':    comp['speed_integral_cm_s'],
    'distance_integral_cm':   comp['distance_integral_cm'],
    'speed_kf_cm_s':          comp['speed_kf_cm_s'],
    'distance_kf_cm':         comp['distance_kf_cm'],
    'speed_cm_s':             comp['speed_cm_s'],
    'distance_live_cm':       comp['distance_live_cm'],
})
travel = load_travel(df)
press = df['air_pressure_pa'].values.astype(float)
a_mean = np.array([r.mean() for r in travel])
slope = pressure_slope(press)
trips, _ = detect_trips(a_mean, slope)

# 每行标注 trip_id（0=静止，1..N=第N趟）
trip_id = np.zeros(len(df), dtype=int)
for i, (s, e) in enumerate(trips):
    trip_id[s:e+1] = i + 1

# 每行 motion_status
motion_status = np.where(
    trip_id > 0,
    '移动中',
    np.where(np.abs(slope) < SLOPE_STATIC_THR, '静止', '过渡中')
)

# 每行 pressure_direction
pressure_dir = []
for i in range(len(df)):
    if trip_id[i] == 0:
        pressure_dir.append('未知')
    elif -slope[i] > 0.5:
        pressure_dir.append('上行')
    elif -slope[i] < -0.5:
        pressure_dir.append('下行')
    else:
        pressure_dir.append('未知')

# 行程最后一行 → '行程结束重置'
for s, e in trips:
    motion_status[e] = '行程结束重置'

# ========== 2. 组装完整数据表 ==========
full = pd.DataFrame({
    'source_row':        np.arange(len(df)) + 2,          # 从2开始（参考Excel约定）
    'utc_time':          df['utc_time'],
    'air_pressure_pa':   df['air_pressure_pa'],
    'trip_id':           trip_id,
    'motion_status':     motion_status,
    'pressure_direction': pressure_dir,
    'kalman_speed_cm_s':      df['speed_kf_cm_s'].astype(int),
    'kalman_displacement_cm': df['distance_kf_cm'].astype(int),
    'naive_speed_cm_s':       df['speed_integral_cm_s'].astype(int),
    'naive_displacement_cm':  df['distance_integral_cm'].astype(int),
    'supplier_speed_cm_s':    df['speed_cm_s'].astype(int),
    'supplier_displacement_cm': df['distance_live_cm'].astype(int),
})

zero_rows = full[full['trip_id'] == 0].copy()
motion_rows = full[full['trip_id'] > 0].copy()

# ========== 3. 准确性评估表 ==========
rows = []
for i, (s, e) in enumerate(trips):
    seg = full.iloc[s:e+1]
    total = len(seg)
    ref_v = seg['supplier_speed_cm_s'].values.astype(float)
    ref_d = seg['supplier_displacement_cm'].values.astype(float)
    # 速度
    naive_v = seg['naive_speed_cm_s'].values.astype(float)
    kf_v = seg['kalman_speed_cm_s'].values.astype(float)
    # 位移（用绝对值比较，因为我们的位移有方向、参考列是 live 累积）
    naive_d = seg['naive_displacement_cm'].values.astype(float)
    kf_d = seg['kalman_displacement_cm'].values.astype(float)

    # 速度准确率：supplier_speed > 0 的行才比较（运动行）
    mask_v = ref_v > 0
    if mask_v.sum() == 0:
        mask_v = np.ones(total, dtype=bool)  # fallback
    naive_v_ok = (np.abs(naive_v - ref_v) <= SPEED_TOL) & mask_v
    kf_v_ok = (np.abs(kf_v - ref_v) <= SPEED_TOL) & mask_v

    # 位移准确率
    naive_d_ok = np.abs(naive_d - ref_d) <= DIST_TOL
    kf_d_ok = np.abs(kf_d - ref_d) <= DIST_TOL

    rows.append({
        '行程ID':                 i + 1,
        '起始行':                 s,
        '结束行':                 e,
        '总行数':                 total,
        '朴素积分速度容差内行数': int(naive_v_ok.sum()),
        '朴素积分速度准确率':     f"{100*naive_v_ok.sum()/mask_v.sum():.1f}%",
        '朴素积分速度中位误差':   f"{np.median(np.abs(naive_v[mask_v]-ref_v[mask_v])):.1f}",
        '卡尔曼速度容差内行数':   int(kf_v_ok.sum()),
        '卡尔曼速度准确率':       f"{100*kf_v_ok.sum()/mask_v.sum():.1f}%",
        '卡尔曼速度中位误差':     f"{np.median(np.abs(kf_v[mask_v]-ref_v[mask_v])):.1f}",
        '朴素积分位移容差内行数': int(naive_d_ok.sum()),
        '朴素积分位移准确率':     f"{100*naive_d_ok.sum()/total:.1f}%",
        '卡尔曼位移容差内行数':   int(kf_d_ok.sum()),
        '卡尔曼位移准确率':       f"{100*kf_d_ok.sum()/total:.1f}%",
        '方向':                   pressure_dir[s],
    })

acc = pd.DataFrame(rows)

# ========== 4. 写 Excel（纯数据，不加任何样式，与参考 Excel 一致） ==========
out = '/workspace/elevator_computed_per_trip.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    full.to_excel(writer, sheet_name='完整数据', index=False)
    acc.to_excel(writer, sheet_name='准确性评估', index=False)
    zero_rows.to_excel(writer, sheet_name='零行程数据', index=False)
    for i, (s, e) in enumerate(trips):
        seg = full.iloc[s:e+1].copy()
        seg.to_excel(writer, sheet_name=f'行程_{i+1}', index=False)
print(f'已输出: {out}')
print(f'  Sheet 数: {2 + len(trips) + 1}')
print(f'  行程数: {len(trips)}')
print(f'\n准确性评估汇总（速度容差 {SPEED_TOL} cm/s，位移容差 {DIST_TOL} cm）:')
print(f"  朴素积分速度准确率:  中位 {np.median([float(r['朴素积分速度准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  卡尔曼速度准确率:    中位 {np.median([float(r['卡尔曼速度准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  朴素积分位移准确率:  中位 {np.median([float(r['朴素积分位移准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  卡尔曼位移准确率:    中位 {np.median([float(r['卡尔曼位移准确率'].rstrip('%')) for r in rows]):.1f}%")
