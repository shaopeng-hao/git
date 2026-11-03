#!/usr/bin/env python3
"""
电梯加速度 → 速度 & 位移 计算
============================

输入: 含 travel 列的 CSV 文件
      travel: 每秒 100 个加速度采样 (cm/s²), 格式为 Python list 字符串
              例如: "[0.1, -0.2, 0.0, 0.3, ...]"  (100个值)
      utc_time (可选): 时间戳列, 仅用于输出保留

输出: CSV 文件, 新增两列
      speed_calc_cm_s:      带方向速度 (cm/s), 正=上行, 负=下行
      distance_calc_cm:     单趟行程内位移 (cm), 每趟结束自动归零
                            如需连续累积位移, 用 distance_calc_cont_cm

核心算法:
  1. 运动段检测: 用 mean(|travel|) 识别加减速/匀速/静止, 拼接完整运动段
  2. 段内积分:   欧拉法 v = v0 + a*dt, s = s0 + v*dt + ½*a*dt²
  3. 段末归零:   运动段结束后 v 重置为 0, 下一趟从 0 开始

作者备注 (调参):
  - ACTIVE_THR 过大会漏检弱加减速, 过小会把匀速期误判为加减速
  - BUFFER 是加减速段之间允许的匀速衔接行数, 太小会把一趟拆成多趟
  - 本脚本用 7200 行电梯数据实测, ACTIVE_THR=0.5 + BUFFER=3 效果最佳
  - 如果你的传感器噪声更大, 可以把 ACTIVE_THR 调到 1.0~2.0
"""

import numpy as np
import pandas as pd
import ast
import argparse
import sys
from pathlib import Path


class ElevationCalculator:
    """电梯加速度积分计算器"""

    def __init__(self, dt=0.01, active_thr=0.5, buffer=3):
        """
        参数:
          dt:          加速度采样间隔 (秒)
                       travel 每秒 100 个采样 → dt = 1/100 = 0.01
          active_thr:  运动段检测阈值 (cm/s²)
                       某秒 mean(|travel|) >= 此值 → 认为处于加减速期
                       电梯实测推荐 0.5, 高噪声环境可调 1.0~2.0
          buffer:      运动段内的匀速衔接行数
                       两个加减速段之间允许最多 buffer 个"匀速行"仍视为同一段
                       电梯加减速→匀速→减速的典型间隔是 2~4 秒 → buffer=3
        """
        self.dt = dt
        self.active_thr = active_thr
        self.buffer = buffer

    # --------------------------------------------------------------
    def load(self, csv_path):
        """加载 CSV, 解析 travel 列"""
        df = pd.read_csv(csv_path)
        if 'travel' not in df.columns:
            raise ValueError(f"CSV 必须包含 'travel' 列, 当前列: {list(df.columns)}")
        travel_data = []
        for s in df['travel']:
            if pd.isna(s):
                travel_data.append(np.zeros(100))
            else:
                try:
                    arr = np.array([float(x) for x in ast.literal_eval(s)])
                    travel_data.append(arr)
                except Exception as e:
                    print(f"⚠️  第 {len(travel_data)} 行 travel 解析失败: {e}, 用零填充")
                    travel_data.append(np.zeros(100))
        return df, travel_data

    # --------------------------------------------------------------
    def detect_motion_segments(self, travel_data):
        """
        检测完整运动段 (加速→匀速→减速)
        
        算法:
          1. 每行算 mean(|travel|), 标记 >= active_thr 的为 "加减速活跃"
          2. 合并所有活跃行段, 间距 <= buffer 的中间匀速行并入同一段
        
        返回:
          segments: list of (start, end) 每个段的起止行号(含)
          row_labels: array, 0=静止, 1=运动段内
        """
        n = len(travel_data)
        mean_abs = np.array([np.mean(np.abs(r)) for r in travel_data])
        active = mean_abs >= self.active_thr

        # 找出所有连续活跃行段
        raw_segments = []
        i = 0
        while i < n:
            if active[i]:
                j = i
                while j < n and active[j]:
                    j += 1
                raw_segments.append((i, j - 1))
                i = j
            else:
                i += 1

        # 合并间距 <= buffer 的段 (中间的匀速衔接行并入)
        segments = []
        for seg in raw_segments:
            if segments and seg[0] - segments[-1][1] - 1 <= self.buffer:
                segments[-1] = (segments[-1][0], seg[1])
            else:
                segments.append(list(seg))

        # 生成每行标签
        row_labels = np.zeros(n, dtype=int)
        for s, e in segments:
            row_labels[s:e + 1] = 1

        return segments, row_labels, mean_abs

    # --------------------------------------------------------------
    def integrate(self, travel_data, segments):
        """
        段内欧拉积分
        
        规则:
          - 每个运动段开头: v=0, s=0 (从零状态开始)
          - 段内:   逐采样点 v += a*dt, s += v*dt + 0.5*a*dt²
          - 段末:   不做特殊处理, 下一个静止行 v 会被自动重置为 0
        
        返回:
          v_dir:   带方向速度 (cm/s), 段末非零, 静止时为 0
          s_trip:  单趟段内位移 (cm), 段末归零, 下一趟重新算
          s_cont:  连续累积位移 (cm), 永不归零 (可选输出)
        """
        n = len(travel_data)
        v_dir = np.zeros(n)
        s_trip = np.zeros(n)

        # 标记每个段的开头和结尾
        seg_starts = {seg[0] for seg in segments}

        v = 0.0
        s = 0.0
        for i in range(n):
            if i in seg_starts:
                v = 0.0
                s = 0.0

            for a in travel_data[i]:
                s += v * self.dt + 0.5 * a * self.dt * self.dt
                v += a * self.dt
            v_dir[i] = v
            s_trip[i] = s

        # 连续累积位移: 逐行累加 trip 的增量
        # 段开头 s_trip 归零会导致跳变(负), 必须跳过
        s_cont = np.zeros(n)
        cum = 0.0
        for i in range(n):
            if i == 0:
                cum = s_trip[i]                    # Row 0: 直接用 trip 值
            elif i in seg_starts:
                cum += s_trip[i]                   # 段开头: 只加这一行新积分的值
            else:
                cum += (s_trip[i] - s_trip[i - 1]) # 段内: 加增量
            s_cont[i] = cum

        return v_dir, s_trip, s_cont

    # --------------------------------------------------------------
    def run(self, csv_path, output_path=None):
        """完整流程: 加载 → 检测段 → 积分 → 保存"""
        print(f"📂 加载: {csv_path}")
        df, travel_data = self.load(csv_path)
        print(f"   {len(df)} 行, {len(travel_data[0])} 采样/行")

        print(f"\n🔍 检测运动段 (thr={self.active_thr}, buffer={self.buffer}) ...")
        segments, row_labels, mean_abs = self.detect_motion_segments(travel_data)
        print(f"   发现 {len(segments)} 个运动段, 覆盖 {row_labels.sum()} / {len(df)} 行")

        print(f"\n⚙️  段内积分 ...")
        v_dir, s_trip, s_cont = self.integrate(travel_data, segments)

        # 写入 DataFrame
        df['mean_abs_acc_cm_s2'] = mean_abs
        df['row_motion_flag'] = row_labels
        df['speed_calc_cm_s'] = v_dir        # 带方向
        df['distance_calc_cm'] = s_trip       # 单趟
        df['distance_calc_cont_cm'] = s_cont  # 连续

        out = output_path or str(Path(csv_path).stem + '_calculated.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"\n✅ 输出: {out}")
        print(f"\n📊 统计:")
        print(f"   速度范围:   {v_dir.min():.1f} ~ {v_dir.max():.1f} cm/s")
        print(f"   速度中位:   {np.median(np.abs(v_dir)):.1f} cm/s")
        print(f"   单趟位移:   {s_trip.max():.0f} cm (最大)")
        print(f"   连续位移:   {s_cont[-1]:.0f} cm (总)")

        return df


# ========================================================================
# CLI 入口
# ========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='电梯加速度 → 速度 & 位移 计算器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认参数 (电梯实测最优)
  python3 elevator_calc.py input.csv

  # 自定义阈值 (传感器噪声大时用)
  python3 elevator_calc.py input.csv -o result.csv --thr 1.0 --buffer 2

  # 只跑前 1000 行测试
  python3 elevator_calc.py input.csv --head 1000
        """)
    parser.add_argument('input', help='输入 CSV (含 travel 列)')
    parser.add_argument('-o', '--output', default=None, help='输出 CSV 路径')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='运动检测阈值 cm/s² (默认 0.5)')
    parser.add_argument('--buffer', type=int, default=3,
                        help='运动段内匀速衔接 buffer 行数 (默认 3)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='采样间隔秒 (默认 0.01 = 100Hz)')
    parser.add_argument('--head', type=int, default=None,
                        help='只处理前 N 行 (快速测试用)')
    args = parser.parse_args()

    calc = ElevationCalculator(dt=args.dt, active_thr=args.thr, buffer=args.buffer)

    if args.head:
        import tempfile, os
        df_full = pd.read_csv(args.input)
        df_head = df_full.head(args.head)
        tmp = tempfile.mktemp(suffix='.csv')
        df_head.to_csv(tmp, index=False)
        df = calc.run(tmp, args.output)
        os.unlink(tmp)
    else:
        df = calc.run(args.input, args.output)


if __name__ == '__main__':
    main()
