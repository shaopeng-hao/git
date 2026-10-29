#!/usr/bin/env python3
"""
电梯数据最终版 v3 —— 运动段检测 + 段内积分 + 段末归零

核心思路:
  1. 先从 travel 识别每个秒是否处于"活跃加减速" (mean_abs >= THR)
  2. 把连续的活跃行 + 中间的匀速行(缓冲) 拼成完整运动段
  3. 段内: 正常积分 (欧拉或KF)
  4. 段结束后: 速度归零, 位置从该段起点开始算 (即单趟位移)
  5. 输出:
     speed_calc: 带方向速度 (cm/s)
     distance_calc: 单趟行程内位移, 段结束时重置为 0 (cm)
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

def load_data(fp):
    df = pd.read_csv(fp)
    tp = []
    for s in df['travel']:
        if pd.isna(s): tp.append(np.zeros(100))
        else: tp.append(np.array([float(x) for x in ast.literal_eval(s)]))
    return df, tp


# ========================================================================
# Step 1: 运动段检测
# ========================================================================
def detect_motion_segments(travel_data, thr=0.5, buffer=2):
    """
    检测完整运动段 (加速→匀速→减速)
    thr: mean_abs 阈值, >= 视为加减速活跃
    buffer: 连续活跃行之间允许多少个非活跃行仍视为同一段(覆盖匀速期)
    
    返回:
      segments: list of (start_idx, end_idx), 每个运动段的起止行号(含)
      row_labels: array, 每行标签: 0=静止, 1=运动段内
    """
    n = len(travel_data)
    # 计算每行 mean_abs
    mean_abs = np.array([np.mean(np.abs(r)) for r in travel_data])
    # 活跃标记
    active = mean_abs >= thr

    # 扩展活跃区间: 允许 buffer 个非活跃行夹在活跃行之间 (匀速衔接)
    # 方法: 先找所有连续活跃段, 然后合并间距 <= buffer 的段
    segments_raw = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            segments_raw.append((i, j - 1))
            i = j
        else:
            i += 1

    # 合并间距 <= buffer 的段 (中间的匀速期并入)
    segments = []
    for seg in segments_raw:
        if segments and seg[0] - segments[-1][1] - 1 <= buffer:
            # 合并
            segments[-1] = (segments[-1][0], seg[1])
        else:
            segments.append(list(seg))

    # 生成每行标签
    row_labels = np.zeros(n, dtype=int)
    for s, e in segments:
        row_labels[s:e+1] = 1

    return segments, row_labels, mean_abs


# ========================================================================
# Step 2: 段内积分 (欧拉法)
# ========================================================================
def integrate_segments_euler(travel_data, segments, dt=0.01):
    """
    每个运动段内部正常积分, 段末速度归零
    返回 v (带方向), s (单趟位移, 段末归零重置)
    """
    n = len(travel_data)
    v = np.zeros(n)
    s = np.zeros(n)

    # 先填充所有静止行 = 0
    # 然后逐段积分
    all_seg_idx = []
    for seg in segments:
        all_seg_idx.extend(range(seg[0], seg[1] + 1))

    # 只有运动段内的行需要积分, 其余都是 0
    # 但是段与段之间的"衔接静止期" v 必须归零, 段开头 v 从 0 开始
    
    v_cum = 0.0
    s_cum = 0.0
    # 先全部填好累计 (跨段归零)
    for i in range(n):
        # 检查这一行是不是新段的开始
        is_new_seg = any(seg[0] == i for seg in segments)
        is_seg_end = any(seg[1] == i for seg in segments)

        if is_new_seg:
            v_cum = 0.0
            s_cum = 0.0

        # 正常积分
        for a in travel_data[i]:
            s_cum += v_cum * dt + 0.5 * a * dt * dt
            v_cum += a * dt
        v[i] = v_cum
        s[i] = s_cum

        if is_seg_end:
            # 段结束, 下一个静止行 v 自动保持为 0 (因为下一次遇到 is_new_seg 就归零)
            pass

    return v, s


# ========================================================================
# Step 2: 段内积分 (KF 三阶)
# ========================================================================
def integrate_segments_kf(travel_data, segments, dt=0.01):
    F = np.array([[1, dt, 0.5*dt*dt], [0, 1, dt], [0, 0, 1]])
    H = np.array([[0, 0, 1]])
    I = np.eye(3)
    Q = np.diag([0.01, 0.1, 0.01**2])
    R = np.array([[2.0**2]])

    n = len(travel_data)
    v = np.zeros(n)
    s = np.zeros(n)
    x = np.array([[0.0], [0.0], [0.0]])
    P = np.diag([100.0, 100.0, 10.0])

    for i in range(n):
        is_new_seg = any(seg[0] == i for seg in segments)
        if is_new_seg:
            x = np.array([[0.0], [0.0], [0.0]])
            P = np.diag([100.0, 100.0, 10.0])

        for a_meas in travel_data[i]:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            y = np.array([[a_meas]]) - H @ x_pred
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ y
            P = (I - K @ H) @ P_pred
        v[i] = x[1, 0]
        s[i] = x[0, 0]

    return v, s


# ========================================================================
# 评估
# ========================================================================
def evaluate_all(name, v, s, speed_o, dist_o):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    vdiff = np.abs(np.abs(v) - speed_o)
    mm = speed_o > 5
    print(f"速度: 中位={np.median(vdiff):.2f}  平均={vdiff.mean():.2f}  <5cm/s={100*(vdiff<5).mean():.1f}%  <10cm/s={100*(vdiff<10).mean():.1f}%")
    if mm.sum() > 0:
        vm = vdiff[mm]
        print(f"  运动中: 中位={np.median(vm):.2f}  平均={vm.mean():.2f}")

    trips, i = [], 0
    while i < len(speed_o):
        if speed_o[i] == 0:
            j = i + 1
            while j < len(speed_o) and speed_o[j] > 0: j += 1
            if j > i + 1:
                trips.append((dist_o[j-1] - dist_o[i], s[j-1] - s[i]))
            i = j
        else: i += 1
    if trips:
        enc = np.array([t[0] for t in trips])
        calc = np.array([t[1] for t in trips])
        err = np.abs(calc - enc)
        rel = err / np.maximum(np.abs(enc), 1) * 100
        print(f"位移({len(trips)}趟): 中位={np.median(err):.1f}cm  平均={err.mean():.1f}cm  相对={rel.mean():.1f}%  >100cm={100*(err>100).mean():.1f}%")
        # 对比 encoder 的位移和我们的位移的线性关系
        if np.std(enc) > 0 and np.std(calc) > 0:
            corr = np.corrcoef(enc, calc)[0, 1]
            print(f"  与 encoder 位移的相关系数: {corr:.4f}")


# ========================================================================
def main():
    fp = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
    df, tp = load_data(fp)
    speed_o = df['speed_cm_s'].values.astype(float)
    dist_o = df['distance_live_cm'].values.astype(float)

    print("=" * 60)
    print("  电梯数据 v3: 运动段检测 + 段内积分")
    print("  只输入 travel + utc_time")
    print("=" * 60)

    # 检测运动段
    segments, row_labels, mean_abs = detect_motion_segments(tp, thr=0.5, buffer=3)
    print(f"\n检测到 {len(segments)} 个运动段")
    print(f"  段长度范围: {[seg[1]-seg[0]+1 for seg in segments]}")
    print(f"  运动段内行数: {row_labels.sum()} / {len(row_labels)}")

    # 段内积分
    print("\n>>> 欧拉法积分 ...")
    v_e, s_e = integrate_segments_euler(tp, segments)

    print(">>> KF三阶积分 ...")
    v_k, s_k = integrate_segments_kf(tp, segments)

    # 评估
    evaluate_all("① 欧拉积分 (段内)", v_e, s_e, speed_o, dist_o)
    evaluate_all("② KF三阶 (段内)", v_k, s_k, speed_o, dist_o)

    # 汇总对比供应商
    print("\n" + "=" * 60)
    print("  供应商 speed_cm_s vs 我们的计算")
    print("=" * 60)
    
    # 速度相关系数
    mask = speed_o > 0
    for name, v in [("欧拉", v_e), ("KF", v_k)]:
        abs_v = np.abs(v)
        corr = np.corrcoef(abs_v[mask], speed_o[mask])[0, 1]
        print(f"  {name} 速度与供应商 speed_cm_s 相关系数 (运动中): {corr:.4f}")

    # 保存
    out = pd.DataFrame({
        'utc_time': df['utc_time'],
        'speed_orig': speed_o,
        'dist_orig': dist_o,
        'mean_abs_acc': mean_abs,
        'row_label_motion': row_labels,
        'v_euler': v_e, 's_euler': s_e,
        'v_kf': v_k, 's_kf': s_k,
    })
    out.to_csv('/workspace/v3_result.csv', index=False)
    print("\n✓ 已保存 v3_result.csv")

    # 可视化 (Row 0~400)
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    n_plot = 400
    x = np.arange(n_plot)

    # 1. mean_abs + 运动段标签
    ax = axes[0]
    ax.plot(x, mean_abs[:n_plot], 'g-', lw=0.8, alpha=0.7, label='mean(|travel|)')
    ax.fill_between(x, 0, 65, where=row_labels[:n_plot]==1, alpha=0.15, color='red', label='检测到的运动段')
    ax.axhline(0.5, color='red', ls='--', lw=0.8, label='thr=0.5')
    ax.set_ylabel('mean(|a|)')
    ax.set_title('(1) 运动检测信号 + 段标签')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2. 速度对比
    ax = axes[1]
    ax.plot(x, speed_o[:n_plot], 'k-', lw=2, label='供应商 speed_cm_s (参考)', zorder=10)
    ax.plot(x, np.abs(v_e[:n_plot]), 'b-', lw=1.2, label='欧拉', alpha=0.9)
    ax.plot(x, np.abs(v_k[:n_plot]), 'r--', lw=1.2, label='KF三阶', alpha=0.9)
    ax.set_ylabel('速度 (cm/s)')
    ax.set_title('(2) 速度对比 (绝对值)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 3. 带方向速度
    ax = axes[2]
    ax.axhline(0, color='black', lw=0.5)
    ax.plot(x, v_e[:n_plot], 'b-', lw=1.2, label='欧拉 (带方向)', alpha=0.85)
    ax.plot(x, v_k[:n_plot], 'r--', lw=1.2, label='KF (带方向)', alpha=0.85)
    ax.set_ylabel('带方向速度')
    ax.set_title('(3) 带方向速度 (+上 -下)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 4. 速度误差
    ax = axes[3]
    err_e = np.abs(np.abs(v_e[:n_plot]) - speed_o[:n_plot])
    err_k = np.abs(np.abs(v_k[:n_plot]) - speed_o[:n_plot])
    ax.plot(x, err_e, 'b-', lw=1, label='欧拉误差', alpha=0.7)
    ax.plot(x, err_k, 'r--', lw=1, label='KF误差', alpha=0.7)
    ax.axhline(10, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('|速度误差|')
    ax.set_yscale('log')
    ax.set_title('(4) 速度误差 (log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 5. 位移对比
    ax = axes[4]
    ax.plot(x, dist_o[:n_plot], 'k-', lw=2, label='供应商 distance_live_cm', zorder=10)
    ax.plot(x, s_e[:n_plot], 'b-', lw=1.2, label='欧拉段内位移', alpha=0.9)
    ax.plot(x, s_k[:n_plot], 'r--', lw=1.2, label='KF段内位移', alpha=0.9)
    ax.set_ylabel('位移 (cm)')
    ax.set_xlabel('行号')
    ax.set_title('(5) 位移对比 (段内单趟, 段末归零)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('/workspace/v3_result.png', dpi=150, bbox_inches='tight')
    print("✓ 已保存 v3_result.png")


if __name__ == '__main__':
    main()
