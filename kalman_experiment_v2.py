#!/usr/bin/env python3
"""
电梯数据对比实验（纯输入版 —— 只依赖 travel + utc_time）

真相澄清：
  实际输入: 只有 travel (每秒100个加速度采样, cm/s²) + utc_time
  speed_cm_s, distance_live_cm: 供应商提供的参考值, 仅用于评测, 不能偷看

四种方案 (全部只用 travel 作为输入):
  ① 直接欧拉积分 (Direct Euler) —— 静止检测用 travel 自身
  ② 一阶低通滤波 + 积分 —— 先平滑加速度再积分
  ③ KF 三阶卡尔曼滤波 (前向实时) —— state=[s, v, a], a 是观测
  ④ KF 三阶 + RTS 平滑 (离线最优) —— 前向KF + 反向平滑

关键修正: 之前 speed_cm_s==0 归零是作弊！现在必须从 travel 本身检测静止。
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据加载
# ============================================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    travel_parsed = []
    for s in df['travel']:
        if pd.isna(s):
            travel_parsed.append(np.zeros(100))
        else:
            arr = np.array([float(x) for x in ast.literal_eval(s)])
            travel_parsed.append(arr)
    return df, travel_parsed


# ============================================================
# 静止检测 (唯一依赖 travel 的判据)
# 100 个采样点的加速度都接近 0 → 静止
# ============================================================
def detect_static(travel_row, thr=0.3):
    """
    travel_row: 该秒 100 个加速度采样
    thr: 阈值 cm/s², 低于此值认为无运动
    返回: True=静止, False=运动
    """
    return np.max(np.abs(travel_row)) < thr


# ============================================================
# 方案①: 直接欧拉积分 (无作弊版)
# ============================================================
def direct_euler(travel_data, dt=0.01, static_thr=0.3):
    n = len(travel_data)
    v = 0.0
    s = 0.0
    v_list = []
    s_list = []

    for i in range(n):
        row = travel_data[i]
        if detect_static(row, thr=static_thr):
            v = 0.0       # 静止 → 强制归零, 消除漂移

        ds = 0.0
        for a in row:
            ds += v * dt + 0.5 * a * dt * dt
            v += a * dt
        s += ds
        v_list.append(v)
        s_list.append(s)

    return np.array(v_list), np.array(s_list)


# ============================================================
# 方案②: 一阶低通滤波 + 积分
# ============================================================
def lowpass_euler(travel_data, dt=0.01, static_thr=0.3, alpha=0.4):
    n = len(travel_data)
    v = 0.0
    s = 0.0
    a_filt = 0.0
    v_list = []
    s_list = []

    for i in range(n):
        row = travel_data[i]
        if detect_static(row, thr=static_thr):
            v = 0.0
            a_filt = 0.0

        ds = 0.0
        for a_raw in row:
            a_filt = alpha * a_raw + (1 - alpha) * a_filt
            ds += v * dt + 0.5 * a_filt * dt * dt
            v += a_filt * dt
        s += ds
        v_list.append(v)
        s_list.append(s)

    return np.array(v_list), np.array(s_list)


# ============================================================
# 方案③: KF 三阶 (前向实时)
# 状态 x = [位置 s, 速度 v, 加速度 a]^T
# 物理模型: 恒加速度 F
# 观测 z = 加速度计读数 (直接观测 a)
# ============================================================
def kf3_forward(travel_data, dt=0.01, static_thr=0.3):
    # 状态转移矩阵 (恒加速度模型)
    F = np.array([
        [1, dt, 0.5*dt*dt],   # s → s + v*dt + 0.5*a*dt²
        [0,  1, dt],           # v → v + a*dt
        [0,  0,  1],           # a → a (假设恒加速度)
    ])
    H = np.array([[0, 0, 1]])    # 观测矩阵: z = a_meas
    I = np.eye(3)

    # 噪声参数 (调参!)
    # Q: 10ms 内加速度的不确定性 (电梯加减速平滑, 取较小值)
    sigma_q_s = 0.1
    sigma_q_v = 0.5
    sigma_q_a = 0.02            # 加速度过程噪声 cm/s² per 10ms
    Q = np.diag([sigma_q_s**2, sigma_q_v**2, sigma_q_a**2])

    # R: 加速度计读数噪声 (静止时测量 σ ≈ 1~2)
    sigma_r = 2.0
    R = np.array([[sigma_r**2]])

    n = len(travel_data)
    # 轨迹保存 (用于后续 RTS 平滑)
    xs = []    # 每个子采样点的 x
    Ps = []    # 每个子采样点的 P
    xs_pred = []
    Ps_pred = []

    v_row_end = []
    s_row_start = []

    x = np.array([[0.0], [0.0], [0.0]])
    P = np.diag([100.0, 100.0, 10.0])

    for i in range(n):
        row = travel_data[i]
        # 静止检测: 强制归零, 重置协方差
        if detect_static(row, thr=static_thr):
            x = np.array([[0.0], [0.0], [0.0]])
            P = np.diag([100.0, 100.0, 10.0])

        s_row_start.append(x[0, 0])

        for a_meas in row:
            # Prediction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            z = np.array([[a_meas]])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (I - K @ H) @ P_pred

            xs.append(x.copy())
            Ps.append(P.copy())
            xs_pred.append(x_pred.copy())
            Ps_pred.append(P_pred.copy())

        v_row_end.append(x[1, 0])

    # 把每个采样点的 x 展开到行级别: 每行的状态取最后一个子采样点
    # 同时返回 100xN 点的轨迹, 供 RTS 使用
    return np.array(v_row_end), xs, Ps, xs_pred, Ps_pred


# ============================================================
# 方案④: KF 三阶 + RTS 反向平滑 (离线最优)
# RTS 公式 (Rauch-Tung-Striebel):
#   x_smooth[k] = x_k^+ + A_k @ (x_smooth[k+1] - x_{k+1}^-)
#   P_smooth[k] = P_k^+ + A_k @ (P_smooth[k+1] - P_{k+1}^-) @ A_k^T
#   A_k = P_k^+ @ F^T @ inv(P_{k+1}^-)
# ============================================================
def rts_smooth(xs, Ps, xs_pred, Ps_pred, F):
    """
    xs, Ps, xs_pred, Ps_pred: 长度为 total_samples (所有行 × 100)
    """
    N = len(xs)
    x_smooth = [None] * N
    P_smooth = [None] * N
    x_smooth[-1] = xs[-1]
    P_smooth[-1] = Ps[-1]

    for k in range(N - 2, -1, -1):
        P_pred_next = Ps_pred[k+1]
        A_k = Ps[k] @ F.T @ np.linalg.inv(P_pred_next)
        x_smooth[k] = xs[k] + A_k @ (x_smooth[k+1] - xs_pred[k+1])
        P_smooth[k] = Ps[k] + A_k @ (P_smooth[k+1] - P_pred_next) @ A_k.T

    return x_smooth, P_smooth


def kf3_rts(travel_data, dt=0.01, static_thr=0.3):
    """
    先跑前向 KF, 再 RTS 反向平滑, 最后从平滑轨迹提取每行的 v 和 s
    """
    v_row_end, xs, Ps, xs_pred, Ps_pred = kf3_forward(
        travel_data, dt=dt, static_thr=static_thr)
    F = np.array([
        [1, dt, 0.5*dt*dt],
        [0,  1, dt],
        [0,  0,  1],
    ])
    x_smooth, P_smooth = rts_smooth(xs, Ps, xs_pred, Ps_pred, F)

    # 从平滑轨迹提取每行末状态
    n_rows = len(travel_data)
    subsamples_per_row = len(travel_data[0])  # 应该是 100
    v_smooth = []
    s_smooth = []

    for i in range(n_rows):
        idx = (i + 1) * subsamples_per_row - 1  # 本行最后一个采样点
        if idx < len(x_smooth):
            v_smooth.append(x_smooth[idx][1, 0])
            s_smooth.append(x_smooth[idx][0, 0])
        else:
            v_smooth.append(xs[-1][1, 0])
            s_smooth.append(xs[-1][0, 0])

    return np.array(v_smooth), np.array(s_smooth)


# ============================================================
# 评估 vs 供应商参考
# ============================================================
def evaluate_speed(name, v_calc, speed_orig):
    v_diff = np.abs(np.abs(v_calc) - speed_orig)
    mask_moving = speed_orig > 5
    n = len(v_diff)

    print(f"\n【{name} - 速度】")
    print(f"  误差 < 3 cm/s:  {(v_diff < 3).sum():>5d} ({100*(v_diff < 3).mean():.1f}%)")
    print(f"  误差 < 5 cm/s:  {(v_diff < 5).sum():>5d} ({100*(v_diff < 5).mean():.1f}%)")
    print(f"  误差 < 10 cm/s: {(v_diff < 10).sum():>5d} ({100*(v_diff < 10).mean():.1f}%)")
    print(f"  平均误差:       {v_diff.mean():.2f} cm/s")
    print(f"  中位误差:       {np.median(v_diff):.2f} cm/s")
    if mask_moving.sum() > 0:
        vm = v_diff[mask_moving]
        print(f"  [运动中] 平均:  {vm.mean():.2f}  中位: {np.median(vm):.2f}")
    return v_diff


def evaluate_distance(name, s_calc, speed_orig, dist_orig):
    # 因为 distance_live_cm 有跳变重置, 不适合逐点比, 用行程增量比
    trips = []
    i = 0
    while i < len(speed_orig):
        if speed_orig[i] == 0:
            j = i + 1
            while j < len(speed_orig) and speed_orig[j] > 0:
                j += 1
            if j > i + 1:
                enc_delta = dist_orig[j-1] - dist_orig[i]
                calc_delta = s_calc[j-1] - s_calc[i]
                trips.append((enc_delta, calc_delta))
            i = j
        else:
            i += 1

    if len(trips) == 0:
        print(f"\n【{name} - 位移】无行程")
        return []
    enc = np.array([t[0] for t in trips])
    calc = np.array([t[1] for t in trips])
    err = np.abs(calc - enc)
    rel = err / np.maximum(np.abs(enc), 1) * 100

    print(f"\n【{name} - 位移】({len(trips)} 趟行程)")
    print(f"  编码器位移范围:  {enc.min():.0f} ~ {enc.max():.0f} cm")
    print(f"  计算位移范围:    {calc.min():.0f} ~ {calc.max():.0f} cm")
    print(f"  平均绝对误差:    {err.mean():.1f} cm")
    print(f"  中位绝对误差:    {np.median(err):.1f} cm")
    print(f"  平均相对误差:    {rel.mean():.1f}%")
    return trips


# ============================================================
# 可视化
# ============================================================
def plot_all(df, results, row_start=0, row_end=300):
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    x = np.arange(row_end - row_start)
    speed_orig = df['speed_cm_s'].values[row_start:row_end]
    dist_orig = df['distance_live_cm'].values[row_start:row_end]

    colors = {
        '直接欧拉': 'steelblue',
        '低通+积分': 'forestgreen',
        'KF 三阶前向': 'crimson',
        'KF + RTS 平滑': 'darkorange',
    }
    markers = ['-', '--', '-.', ':']

    # 子图1: 速度对比
    ax = axes[0]
    ax.plot(x, speed_orig, 'k-', lw=2, label='供应商 speed_cm_s (参考)', zorder=10)
    for (name, (v, s)), mk in zip(results.items(), markers):
        ax.plot(x, np.abs(v[row_start:row_end]), color=colors[name], linestyle=mk,
                lw=1.3, label=name, alpha=0.85)
    ax.set_ylabel('速度 (cm/s)')
    ax.set_title('(上半部) 速度对比')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 子图2: 速度误差
    ax = axes[1]
    for (name, (v, s)), mk in zip(results.items(), markers):
        err = np.abs(np.abs(v[row_start:row_end]) - speed_orig)
        ax.plot(x, err, color=colors[name], linestyle=mk, label=name, alpha=0.8)
    ax.axhline(10, color='gray', ls=':', alpha=0.5, label='10 cm/s 线')
    ax.set_ylabel('|速度误差| (cm/s)')
    ax.set_title('(上半部) 速度误差 (log)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 子图3: 位移对比
    ax = axes[2]
    ax.plot(x, dist_orig, 'k-', lw=2, label='供应商 distance_live_cm (参考)', zorder=10)
    for (name, (v, s)), mk in zip(results.items(), markers):
        ax.plot(x, s[row_start:row_end], color=colors[name], linestyle=mk,
                lw=1.3, label=name, alpha=0.85)
    ax.set_ylabel('位移 (cm)')
    ax.set_title('(下半部) 位移对比')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 子图4: 带方向速度 + 原始加速度信号
    ax = axes[3]
    ax.axhline(0, color='black', lw=0.5)
    for (name, (v, s)), mk in zip(results.items(), markers):
        ax.plot(x, v[row_start:row_end], color=colors[name], linestyle=mk,
                lw=1.3, label=name, alpha=0.8)
    ax.set_ylabel('带方向速度 (cm/s)')
    ax.set_xlabel('行号')
    ax.set_title('(下半部) 带方向速度')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/kalman_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 对比图已保存: /workspace/kalman_comparison_v2.png")


# ============================================================
# 主程序
# ============================================================
def main():
    filepath = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
    print("=" * 65)
    print("  电梯数据对比实验 v2")
    print("  ★ 只依赖 travel + utc_time, 不偷看供应商 speed/distance")
    print("  ★ 静止检测: max(|travel|) < 0.3 cm/s²")
    print("=" * 65)

    df, travel_parsed = load_data(filepath)
    speed_orig = df['speed_cm_s'].values.astype(float)
    dist_orig = df['distance_live_cm'].values.astype(float)
    print(f"\n数据: {len(df)} 行, {df['utc_time'].iloc[0]} ~ {df['utc_time'].iloc[-1]}")

    # ---- ① 直接欧拉 ----
    print("\n>>> ① 直接欧拉积分 ...")
    v1, s1 = direct_euler(travel_parsed, static_thr=0.3)

    # ---- ② 低通+积分 ----
    print(">>> ② 低通滤波 + 积分 ...")
    v2, s2 = lowpass_euler(travel_parsed, static_thr=0.3, alpha=0.4)

    # ---- ③ KF 前向 ----
    print(">>> ③ KF 三阶 (前向实时) ...")
    v3, xs, Ps, xs_pred, Ps_pred = kf3_forward(travel_parsed, static_thr=0.3)

    # ---- ④ KF + RTS ----
    print(">>> ④ KF 三阶 + RTS 反向平滑 (离线最优) ...")
    v4, s4 = kf3_rts(travel_parsed, static_thr=0.3)

    results = {
        '直接欧拉': (v1, s1),
        '低通+积分': (v2, s2),
        'KF 三阶前向': (v3, None),   # s 在评测时算
        'KF + RTS 平滑': (v4, s4),
    }

    # ---- 速度评测 ----
    print("\n" + "=" * 65)
    print("  速度 vs 供应商 speed_cm_s")
    print("=" * 65)
    err_v_all = {}
    for name, (v, s) in results.items():
        err_v_all[name] = evaluate_speed(name, v, speed_orig)

    # ---- 位移评测 ----
    print("\n" + "=" * 65)
    print("  位移 vs 供应商 distance_live_cm (行程增量比)")
    print("=" * 65)
    trips_all = {}
    # 计算 KF 前向的位移 (从轨迹 xs 提取)
    subs = len(travel_parsed[0])
    s3_from_traj = np.array([
        xs[(i+1)*subs - 1][0, 0] if (i+1)*subs - 1 < len(xs) else xs[-1][0, 0]
        for i in range(len(travel_parsed))
    ])
    results['KF 三阶前向'] = (v3, s3_from_traj)
    for name, (v, s) in results.items():
        trips_all[name] = evaluate_distance(name, s, speed_orig, dist_orig)

    # ---- 汇总 ----
    print("\n" + "=" * 65)
    print("  汇总表")
    print("=" * 65)
    print(f"\n{'方案':<20} {'速度中位误':>10} {'速度<5cm/s':>12} {'行程中位误':>12} {'行程相对误':>12}")
    print("-" * 66)
    for name in ['直接欧拉', '低通+积分', 'KF 三阶前向', 'KF + RTS 平滑']:
        v_diff = err_v_all[name]
        trips = trips_all[name]
        if trips:
            enc = np.array([t[0] for t in trips])
            calc = np.array([t[1] for t in trips])
            err = np.abs(calc - enc)
            rel = err / np.maximum(np.abs(enc), 1) * 100
            print(f"{name:<20} {np.median(v_diff):>10.2f} {100*(v_diff<5).mean():>11.1f}% {np.median(err):>11.1f} cm {rel.mean():>11.1f}%")
        else:
            print(f"{name:<20} {np.median(v_diff):>10.2f} {100*(v_diff<5).mean():>11.1f}% {'---':>12} {'---':>12}")

    # ---- 可视化 ----
    plot_all(df, results, row_start=0, row_end=300)

    # ---- 保存 ----
    out = pd.DataFrame({
        'utc_time': df['utc_time'],
        'speed_orig': df['speed_cm_s'],
        'dist_orig': df['distance_live_cm'],
        'v_direct': v1,  's_direct': s1,
        'v_lowpass': v2, 's_lowpass': s2,
        'v_kf': v3,      's_kf': s3_from_traj,
        'v_rts': v4,     's_rts': s4,
    })
    out.to_csv('/workspace/kalman_comparison_v2.csv', index=False, encoding='utf-8')
    print("\n✓ 数据已保存: /workspace/kalman_comparison_v2.csv")


if __name__ == '__main__':
    main()
