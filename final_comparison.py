#!/usr/bin/env python3
"""
电梯数据最终对比: 纯惯性积分 vs 运动窗口积分 vs KF vs RTS
★ 只依赖 travel + utc_time
★ 核心洞察: 电梯运动 = 加减速期积分 + 匀速/静止期保持
★ 静止/匀速检测: mean_abs(travel) < THRESH
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

# ========================================================================
def load_data(fp):
    df = pd.read_csv(fp)
    tp = []
    for s in df['travel']:
        if pd.isna(s): tp.append(np.zeros(100))
        else: tp.append(np.array([float(x) for x in ast.literal_eval(s)]))
    return df, tp

# ========================================================================
def is_active(travel_row, thr=0.5):
    """
    判断该行是否处于加减速期 (需要积分)
    mean(|travel|) >= thr → 加减速, 需要积分
    mean(|travel|) < thr → 静止或匀速, 停止积分, 保持状态
    """
    return np.mean(np.abs(travel_row)) >= thr

# ========================================================================
# 方案 A: 直接积分 + 运动窗口 (只在加减速期积分)
# ========================================================================
def euler_active_only(travel_data, dt=0.01, active_thr=0.5):
    v, s = 0.0, 0.0
    v_list, s_list = [], []
    for row in travel_data:
        if not is_active(row, active_thr):
            # 静止/匀速 → 保持, 速度归零消漂移
            v = 0.0
        else:
            for a in row:
                s += v * dt + 0.5 * a * dt * dt
                v += a * dt
        v_list.append(v)
        s_list.append(s)
    return np.array(v_list), np.array(s_list)

# ========================================================================
# 方案 B: 直接积分 (全程积分, 有漂移)
# ========================================================================
def euler_full(travel_data, dt=0.01):
    v, s = 0.0, 0.0
    v_list, s_list = [], []
    for row in travel_data:
        for a in row:
            s += v * dt + 0.5 * a * dt * dt
            v += a * dt
        v_list.append(v)
        s_list.append(s)
    return np.array(v_list), np.array(s_list)

# ========================================================================
# 方案 C: KF 三阶前向 + 运动窗口重置
# ========================================================================
def kf3_active_only(travel_data, dt=0.01, active_thr=0.5):
    F = np.array([[1, dt, 0.5*dt*dt], [0, 1, dt], [0, 0, 1]])
    H = np.array([[0, 0, 1]])
    I = np.eye(3)
    # 噪声: 加速度计在运动时噪声 σ≈1~2 cm/s²
    Q = np.diag([0.01, 0.1, 0.01**2])
    R = np.array([[2.0**2]])

    x = np.array([[0.0], [0.0], [0.0]])
    P = np.diag([100.0, 100.0, 10.0])
    v_list, s_list = [], []

    for row in travel_data:
        if not is_active(row, active_thr):
            # 静止/匀速 → 归零, 重置协方差
            x = np.array([[0.0], [0.0], [0.0]])
            P = np.diag([100.0, 100.0, 10.0])
        else:
            for a_meas in row:
                # Predict
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                # Update (观测加速度)
                y = np.array([[a_meas]]) - H @ x_pred
                K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
                x = x_pred + K @ y
                P = (I - K @ H) @ P_pred
        v_list.append(x[1, 0])
        s_list.append(x[0, 0])

    return np.array(v_list), np.array(s_list)

# ========================================================================
# 方案 D: 全程 KF (无窗口重置)
# ========================================================================
def kf3_full(travel_data, dt=0.01):
    F = np.array([[1, dt, 0.5*dt*dt], [0, 1, dt], [0, 0, 1]])
    H = np.array([[0, 0, 1]])
    I = np.eye(3)
    Q = np.diag([0.01, 0.1, 0.01**2])
    R = np.array([[2.0**2]])
    x = np.array([[0.0], [0.0], [0.0]])
    P = np.diag([100.0, 100.0, 10.0])
    v_list, s_list = [], []
    for row in travel_data:
        for a_meas in row:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            y = np.array([[a_meas]]) - H @ x_pred
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ y
            P = (I - K @ H) @ P_pred
        v_list.append(x[1, 0])
        s_list.append(x[0, 0])
    return np.array(v_list), np.array(s_list)

# ========================================================================
# 评估
# ========================================================================
def evaluate(name, v, s, speed_o, dist_o):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    vdiff = np.abs(np.abs(v) - speed_o)
    mm = speed_o > 5
    print(f"【速度】中位={np.median(vdiff):.2f} 平均={vdiff.mean():.2f}")
    print(f"  <5cm/s: {100*(vdiff<5).mean():.1f}%  <10cm/s: {100*(vdiff<10).mean():.1f}%")
    if mm.sum() > 0:
        vm = vdiff[mm]
        print(f"  [运动中] 中位={np.median(vm):.2f}  平均={vm.mean():.2f}")

    # 行程位移
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
        print(f"【位移】({len(trips)}趟) 中位={np.median(err):.1f}cm  平均={err.mean():.1f}cm  相对={rel.mean():.1f}%")
        # 大误差趟数
        big = err > 100
        if big.sum() > 0:
            print(f"  大误差(>100cm): {big.sum()} 趟 ({100*big.sum()/len(trips):.1f}%)")

# ========================================================================
def main():
    fp = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
    df, tp = load_data(fp)
    speed_o = df['speed_cm_s'].values.astype(float)
    dist_o = df['distance_live_cm'].values.astype(float)

    # 分析 mean_abs 分布
    mean_abs = np.array([np.mean(np.abs(r)) for r in tp])
    print("=" * 65)
    print("  电梯数据对比 (纯 travel + utc_time 输入)")
    print(f"  mean_abs 分布: min={mean_abs.min():.3f} median={np.median(mean_abs):.3f} p90={np.percentile(mean_abs,90):.3f} p99={np.percentile(mean_abs,99):.1f}")
    print("=" * 65)

    # 跑四种方案
    vA, sA = euler_active_only(tp, active_thr=0.5)
    vB, sB = euler_full(tp)
    vC, sC = kf3_active_only(tp, active_thr=0.5)
    vD, sD = kf3_full(tp)

    # 评估
    evaluate("A: 直接积分 + 运动窗口 (mean_abs<0.5 停止)", vA, sA, speed_o, dist_o)
    evaluate("B: 直接积分 (全程, 有漂移)", vB, sB, speed_o, dist_o)
    evaluate("C: KF三阶 + 运动窗口重置", vC, sC, speed_o, dist_o)
    evaluate("D: KF三阶 (全程, 无重置)", vD, sD, speed_o, dist_o)

    # 汇总
    print("\n" + "=" * 65)
    print("  最终汇总")
    print("=" * 65)
    print(f"\n{'方案':<28} {'速度中位':>8} {'速度<5%':>8} {'位移中位':>10} {'位移相对':>10}")
    print("-" * 65)
    for name, v, s in [("A 直接积分+窗口", vA, sA), ("B 直接积分全程", vB, sB),
                       ("C KF+窗口重置", vC, sC), ("D KF全程", vD, sD)]:
        vd = np.abs(np.abs(v) - speed_o)
        vmed = np.median(vd)
        v5 = 100*(vd<5).mean()
        trips, i = [], 0
        while i < len(speed_o):
            if speed_o[i] == 0:
                j = i+1
                while j < len(speed_o) and speed_o[j] > 0: j += 1
                if j > i+1: trips.append((dist_o[j-1]-dist_o[i], s[j-1]-s[i]))
                i = j
            else: i += 1
        if trips:
            enc = np.array([t[0] for t in trips])
            calc = np.array([t[1] for t in trips])
            err = np.abs(calc - enc)
            rel = err / np.maximum(np.abs(enc), 1) * 100
            print(f"{name:<28} {vmed:>7.2f} {v5:>7.1f}% {np.median(err):>9.1f}cm {rel.mean():>9.1f}%")

    # 保存
    out = pd.DataFrame({
        'utc_time': df['utc_time'],
        'speed_orig': speed_o,
        'dist_orig': dist_o,
        'mean_abs_acc': mean_abs,
        'v_A_euler_win': vA, 's_A_euler_win': sA,
        'v_B_euler_full': vB, 's_B_euler_full': sB,
        'v_C_kf_win': vC, 's_C_kf_win': sC,
        'v_D_kf_full': vD, 's_D_kf_full': sD,
    })
    out.to_csv('/workspace/final_comparison.csv', index=False)
    print("\n✓ 已保存 final_comparison.csv")

    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    x = np.arange(300)
    speed_o_300 = speed_o[:300]
    dist_o_300 = dist_o[:300]
    mean_abs_300 = mean_abs[:300]

    # 子图1: mean_abs (运动检测信号)
    ax = axes[0]
    ax.plot(x, mean_abs_300, 'g-', lw=1, label='mean(|travel|)', alpha=0.7)
    ax.axhline(0.5, color='red', ls='--', label='阈值 0.5 cm/s²')
    ax.set_ylabel('mean(|a|) (cm/s²)')
    ax.set_title('(1) 运动检测信号')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图2: 速度对比
    ax = axes[1]
    ax.plot(x, speed_o_300, 'k-', lw=2, label='供应商 speed_cm_s (参考)', zorder=10)
    ax.plot(x, np.abs(vA[:300]), 'b-', lw=1.5, label='A: 欧拉+窗口', alpha=0.9)
    ax.plot(x, np.abs(vC[:300]), 'r--', lw=1.5, label='C: KF+窗口', alpha=0.9)
    ax.set_ylabel('速度 (cm/s)')
    ax.set_title('(2) 速度对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图3: 位移对比
    ax = axes[2]
    ax.plot(x, dist_o_300, 'k-', lw=2, label='供应商 distance_live_cm (参考)', zorder=10)
    ax.plot(x, sA[:300], 'b-', lw=1.5, label='A: 欧拉+窗口', alpha=0.9)
    ax.plot(x, sC[:300], 'r--', lw=1.5, label='C: KF+窗口', alpha=0.9)
    ax.set_ylabel('位移 (cm)')
    ax.set_xlabel('行号')
    ax.set_title('(3) 位移对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/final_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 已保存 final_comparison.png")

if __name__ == '__main__':
    main()
