#!/usr/bin/env python3
"""
把 elevator_computed.csv 按段拆成多 sheet 的 Excel，按参考 Excel 的列布局，
并计算每一行（及每一行所在趟）的准确率。
"""
import sys, ast
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

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

# ========== 4. 写 Excel ==========
wb = Workbook()
wb.remove(wb.active)

HEADER_FILL = PatternFill('solid', fgColor='1F4E78')
HEADER_FONT = Font(bold=True, color='FFFFFF', name='Arial', size=11)
BODY_FONT = Font(name='Arial', size=10)
ZEBRA1 = PatternFill('solid', fgColor='FFFFFF')
ZEBRA2 = PatternFill('solid', fgColor='F7F9FC')
THIN = Side(style='thin', color='D9DEE7')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal='center', vertical='center')

def write_df(ws, df_in):
    for c_idx, col in enumerate(df_in.columns, 1):
        cell = ws.cell(1, c_idx, col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = CENTER
        cell.border = BORDER
    for r_idx, row in enumerate(df_in.itertuples(index=False), 2):
        fill = ZEBRA1 if (r_idx - 2) % 2 == 0 else ZEBRA2
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(r_idx, c_idx, val)
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.fill = fill
            cell.alignment = CENTER
    # 列宽
    for c_idx, col in enumerate(df_in.columns, 1):
        max_len = max(len(str(col)), df_in[col].astype(str).str.len().max())
        ws.column_dimensions[chr(64 + c_idx) if c_idx <= 26 else 'A' + chr(64 + c_idx - 26)].width = min(max_len + 3, 25)

# 4.1 完整数据
ws = wb.create_sheet('完整数据')
write_df(ws, full)

# 4.2 准确性评估
ws = wb.create_sheet('准确性评估')
write_df(ws, acc)

# 4.3 零行程数据
ws = wb.create_sheet('零行程数据')
write_df(ws, zero_rows)

# 4.4 每趟 sheet
for i, (s, e) in enumerate(trips):
    sheet_name = f'行程_{i+1}'
    ws = wb.create_sheet(sheet_name)
    seg = full.iloc[s:e+1].copy()
    # 在每趟 sheet 右侧追加准确率信息行（只读展示）
    row = rows[i]
    summary = pd.DataFrame([
        ['行程ID', row['行程ID']],
        ['总行数', row['总行数']],
        ['方向', row['方向']],
        ['朴素积分速度准确率', row['朴素积分速度准确率']],
        ['朴素积分速度中位误差', row['朴素积分速度中位误差']],
        ['卡尔曼速度准确率', row['卡尔曼速度准确率']],
        ['卡尔曼速度中位误差', row['卡尔曼速度中位误差']],
        ['朴素积分位移准确率', row['朴素积分位移准确率']],
        ['卡尔曼位移准确率', row['卡尔曼位移准确率']],
    ], columns=['指标', '值'])
    write_df(ws, seg)
    # 追加摘要区（E2 起始列）
    start_col = len(seg.columns) + 3
    for r_idx, (k, v) in enumerate(summary.itertuples(index=False), 2):
        c = ws.cell(r_idx, start_col, k)
        c.font = Font(bold=True, name='Arial', size=10)
        c.alignment = Alignment(horizontal='left', vertical='center')
        c = ws.cell(r_idx, start_col + 1, v)
        c.font = Font(name='Arial', size=10)
        c.alignment = Alignment(horizontal='left', vertical='center')
    ws.column_dimensions[chr(64 + start_col) if start_col <= 26 else 'A' + chr(64 + start_col - 26)].width = 22
    ws.column_dimensions[chr(64 + start_col + 1) if start_col + 1 <= 26 else 'A' + chr(64 + start_col + 1 - 26)].width = 20

out = '/workspace/elevator_computed_per_trip.xlsx'
wb.save(out)
print(f'已输出: {out}')
print(f'  Sheet 数: {len(wb.sheetnames)}')
print(f'  行程数: {len(trips)}')
print(f'\n准确性评估汇总（速度容差 {SPEED_TOL} cm/s，位移容差 {DIST_TOL} cm）:')
print(f"  朴素积分速度准确率:  中位 {np.median([float(r['朴素积分速度准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  卡尔曼速度准确率:    中位 {np.median([float(r['卡尔曼速度准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  朴素积分位移准确率:  中位 {np.median([float(r['朴素积分位移准确率'].rstrip('%')) for r in rows]):.1f}%")
print(f"  卡尔曼位移准确率:    中位 {np.median([float(r['卡尔曼位移准确率'].rstrip('%')) for r in rows]):.1f}%")
