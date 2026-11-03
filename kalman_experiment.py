#!/usr/bin/env python3
"""
电梯数据对比实验: 直接积分 vs 卡尔曼滤波 vs 低通+积分

输入: 只有 travel (每秒100个加速度采样, cm/s²)
输出: 速度和位移
基准: speed_cm_s (绝对值), distance_live_cm (编码器位置) — 仅用于验证

三种方案:
  ① 直接积分 (Direct Euler)
  ② KF 三阶卡尔曼滤波 (位置+速度+加速度) — 加速度作为观测
  ③ 低通滤波 + 积分 (Low-pass filter + Euler)
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

# 设置中文字体
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
            travel_parsed.append(np.array([float(x) for x in ast.literal_eval(s)]))
    return df, travel_parsed


# ============================================================
# 方案①: 直接欧拉积分 (每10ms一个采样)
# ============================================================
def direct_euler_integrate(travel_data, speed_orig, dt=0.01):
    """
    纯欧拉法积分
    当原始 speed==0 时强制归零
    """
    n = len(travel_data)
    v = 0.0
    v_list = []      # 行末带方向速度
    s_list = []      # 连续累积位移
    delta_s_list = []
    s = 0.0

    for i in range(n):
        if speed_orig[i] == 0:
            v = 0.0
        a_list = travel_data[i]
        ds = 0.0
        for a in a_list:
            ds += v * dt + 0.5 * a * dt * dt
            v += a * dt
        s += ds
        v_list.append(v)
        s_list.append(s)
        delta_s_list.append(ds)

    return np.array(v_list), np.array(s_list), np.array(delta_s_list)


# ============================================================
# 方案②: KF 三阶卡尔曼滤波
# 状态 x = [位置, 速度, 加速度]^T
# 物理模型: s = s0 + v0*t + 0.5*a*t^2, v = v0 + a*t, a = a0 (恒加速度)
# 观测 z = 加速度计读数 (直接观测加速度)
# ============================================================
def kalman_filter_3rd_order(travel_data, speed_orig, dt=0.01):
    """
    三阶卡尔曼滤波: state = [s, v, a]
    - 状态转移: 恒加速度模型
    - 观测: 加速度计读数
    - 100Hz 运行 (每个采样点都跑一次 KF)
    """
    # 状态转移矩阵 (恒加速度模型)
    F = np.array([
        [1, dt, 0.5 * dt * dt],   # s_next = s + v*dt + 0.5*a*dt²
        [0,  1, dt],              # v_next = v + a*dt
        [0,  0,  1],              # a_next = a (假设加速度恒定)
    ])

    # 观测矩阵: 我们观测到的就是加速度 a
    H = np.array([[0, 0, 1]])

    # === 噪声参数 (调参!) ===
    # 过程噪声 Q: 加速度在 10ms 内能变多少
    # 如果加速度计很灵敏, Q 可以小; 电梯加减速平缓, Q 也应该小
    sigma_q_acc = 5.0 * dt    # 加速度过程噪声 cm/s² per 10ms → ~0.05
    Q = np.diag([0.01, 0.1, sigma_q_acc**2])  # [s噪声, v噪声, a噪声]

    # 观测噪声 R: 加速度计读数的方差
    # 先看静止时 travel 的方差
    # 假设静止时加速度噪声 σ ≈ 1~3 cm/s²
    sigma_r = 2.0   # cm/s²
    R = np.array([[sigma_r ** 2]])

    # 初始化
    x = np.array([[0.0], [0.0], [0.0]])   # [位置, 速度, 加速度]
    P = np.diag([100.0, 100.0, 10.0])     # 初始大不确定性

    n = len(travel_data)
    v_row_end = []
    s_cont = []
    s_total = 0.0

    for i in range(n):
        # 如果原始 speed==0, 强制归零 (消除漂移)
        if speed_orig[i] == 0:
            x[0, 0] = 0.0
            x[1, 0] = 0.0
            P = np.diag([100.0, 100.0, 10.0])

        a_measurements = travel_data[i]
        row_s_before = x[0, 0]  # 行开始时的位置

        for a_meas in a_measurements:
            # --- Prediction ---
            x = F @ x
            P = F @ P @ F.T + Q

            # --- Update (用加速度观测修正) ---
            z = np.array([[a_meas]])
            y = z - H @ x                    # 残差: a_meas - a_pred
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(3) - K @ H) @ P

        v_row_end.append(x[1, 0])
        s_total += (x[0, 0] - row_s_before)
        s_cont.append(x[0, 0])

    return np.array(v_row_end), np.array(s_cont), None


# ============================================================
# 方案③: 低通滤波 + 积分
# ============================================================
def lowpass_integrate(travel_data, speed_orig, dt=0.01, alpha=0.3):
    """
    一阶低通滤波后再积分
    alpha: 平滑系数, 0~1, 越小越平滑但延迟越大
    """
    n = len(travel_data)
    v = 0.0
    v_list = []
    s_list = []
    s = 0.0
    a_filtered = 0.0  # 上一次的滤波值

    for i in range(n):
        if speed_orig[i] == 0:
            v = 0.0
            a_filtered = 0.0

        a_list = travel_data[i]
        ds = 0.0
        for a_raw in a_list:
            # 低通滤波: a_filtered = α*a_raw + (1-α)*a_filtered
            a_filtered = alpha * a_raw + (1 - alpha) * a_filtered
            ds += v * dt + 0.5 * a_filtered * dt * dt
            v += a_filtered * dt
        s += ds
        v_list.append(v)
        s_list.append(s)

    return np.array(v_list), np.array(s_list), None


# ============================================================
# 评估函数
# ============================================================
def evaluate(name, v_calc, s_calc, speed_orig, dist_orig):
    """计算速度和位移的误差统计"""
    # 速度评估 (只看标量)
    v_diff = np.abs(np.abs(v_calc) - speed_orig)
    mask_moving = speed_orig > 10

    # 位移评估 (整体相关 + 行程相关)
    # 由于编码器 distance_live_cm 有跳变重置, 不适合直接比较绝对值
    # 改用行程内的位移变化来评估
    trip_diffs = []
    speeds = speed_orig
    i = 0
    while i < len(speeds):
        if speeds[i] == 0:
            j = i + 1
            while j < len(speeds) and speeds[j] > 0:
                j += 1
            if j > i + 1:
                # 一趟行程: row i → row j-1
                # 编码器位移变化
                enc_start = dist_orig[i]
                enc_end = dist_orig[j-1]
                enc_delta = enc_end - enc_start
                # 我们计算的位移变化
                calc_delta = s_calc[j-1] - s_calc[i]
                trip_diffs.append((enc_delta, calc_delta))
            i = j
        else:
            i += 1

    enc_deltas = np.array([d[0] for d in trip_diffs])
    calc_deltas = np.array([d[1] for d in trip_diffs])
    trip_err = np.abs(calc_deltas - enc_deltas)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"【速度 vs speed_cm_s】")
    print(f"  误差 < 3 cm/s:  {(v_diff < 3).sum():>5d} ({100*(v_diff < 3).mean():.1f}%)")
    print(f"  误差 < 5 cm/s:  {(v_diff < 5).sum():>5d} ({100*(v_diff < 5).mean():.1f}%)")
    print(f"  误差 < 10 cm/s: {(v_diff < 10).sum():>5d} ({100*(v_diff < 10).mean():.1f}%)")
    print(f"  平均误差:       {v_diff.mean():.2f} cm/s")
    print(f"  中位误差:       {np.median(v_diff):.2f} cm/s")
    if mask_moving.sum() > 0:
        v_diff_m = v_diff[mask_moving]
        print(f"  [运动中] 平均:  {v_diff_m.mean():.2f}  中位: {np.median(v_diff_m):.2f}")

    print(f"\n【单趟行程位移 vs 编码器】({len(trip_diffs)} 趟)")
    if len(trip_diffs) > 0:
        print(f"  编码器位移范围:  {enc_deltas.min():.0f} ~ {enc_deltas.max():.0f} cm")
        print(f"  计算位移范围:    {calc_deltas.min():.0f} ~ {calc_deltas.max():.0f} cm")
        print(f"  平均绝对误差:    {trip_err.mean():.1f} cm")
        print(f"  中位绝对误差:    {np.median(trip_err):.1f} cm")
        # 相对误差
        denom = np.maximum(np.abs(enc_deltas), 1)
        rel_err = trip_err / denom * 100
        print(f"  平均相对误差:    {rel_err.mean():.1f}%")

    return v_diff, trip_diffs


# ============================================================
# 可视化对比
# ============================================================
def plot_comparison(df, results, row_start=0, row_end=200):
    """绘制速度和位移对比图"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    t = df['utc_time'].values[row_start:row_end]
    x = np.arange(row_end - row_start)

    # 原始数据
    speed_orig = df['speed_cm_s'].values[row_start:row_end]
    dist_orig = df['distance_live_cm'].values[row_start:row_end]

    # ---- 子图1: 速度对比 ----
    ax = axes[0]
    ax.plot(x, speed_orig, 'k-', label='原始 speed_cm_s (参考)', linewidth=2)
    colors = {'直接积分': 'b', 'KF三阶': 'r', '低通+积分': 'g'}
    linestyles = {'直接积分': '-', 'KF三阶': '--', '低通+积分': '-.'}
    for name, (v, s, _) in results.items():
        ax.plot(x, np.abs(v[row_start:row_end]),
                color=colors[name], linestyle=linestyles[name],
                label=f'{name}', alpha=0.85)
    ax.set_ylabel('速度 (cm/s)')
    ax.set_title(f'速度对比 (Row {row_start}~{row_end})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图2: 速度误差 ----
    ax = axes[1]
    for name, (v, s, _) in results.items():
        err = np.abs(np.abs(v[row_start:row_end]) - speed_orig)
        ax.plot(x, err, color=colors[name], linestyle=linestyles[name],
                label=f'{name} |速度误差|', alpha=0.8)
    ax.set_ylabel('|速度误差| (cm/s)')
    ax.set_title('速度误差')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # ---- 子图3: 位移对比 ----
    ax = axes[2]
    ax.plot(x, dist_orig, 'k-', label='原始 distance_live_cm (编码器)', linewidth=2)
    for name, (v, s, _) in results.items():
        ax.plot(x, s[row_start:row_end],
                color=colors[name], linestyle=linestyles[name],
                label=f'{name}', alpha=0.85)
    ax.set_ylabel('位移 (cm)')
    ax.set_title(f'位移对比 (注意: 编码器是跳变值, 计算值是连续积分)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图4: 带方向速度 ----
    ax = axes[3]
    ax.axhline(0, color='black', linewidth=0.5)
    for name, (v, s, _) in results.items():
        ax.plot(x, v[row_start:row_end],
                color=colors[name], linestyle=linestyles[name],
                label=f'{name}', alpha=0.8)
    ax.set_ylabel('带方向速度 (cm/s)')
    ax.set_xlabel('行号')
    ax.set_title('带方向速度 (正=上行, 负=下行)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/kalman_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ 对比图已保存: /workspace/kalman_comparison.png")
    plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    filepath = '/workspace/.uploads/46f60c48-5eb6-4fb1-bd39-b68871edd22d_nke_data_merged--42844610.csv'
    print("=" * 60)
    print("  电梯数据对比实验")
    print("  输入: 只有 travel (加速度)")
    print("  基准: speed_cm_s, distance_live_cm (仅用于验证)")
    print("=" * 60)

    # 加载
    df, travel_parsed = load_data(filepath)
    speed_orig = df['speed_cm_s'].values.astype(float)
    dist_orig = df['distance_live_cm'].values.astype(float)
    print(f"\n数据: {len(df)} 行, 时间 {df['utc_time'].iloc[0]} ~ {df['utc_time'].iloc[-1]}")

    # ---- ① 直接积分 ----
    print("\n>>> 方案①: 直接欧拉积分 ...")
    v1, s1, _ = direct_euler_integrate(travel_parsed, speed_orig)
    err1_v, trips1 = evaluate("① 直接欧拉积分", v1, s1, speed_orig, dist_orig)

    # ---- ② KF 三阶 ----
    print("\n>>> 方案②: KF 三阶卡尔曼滤波 ...")
    v2, s2, _ = kalman_filter_3rd_order(travel_parsed, speed_orig)
    err2_v, trips2 = evaluate("② KF 三阶卡尔曼滤波", v2, s2, speed_orig, dist_orig)

    # ---- ③ 低通+积分 ----
    print("\n>>> 方案③: 低通滤波 + 积分 ...")
    v3, s3, _ = lowpass_integrate(travel_parsed, speed_orig, alpha=0.4)
    err3_v, trips3 = evaluate("③ 低通滤波 + 积分", v3, s3, speed_orig, dist_orig)

    # ---- 汇总对比 ----
    print("\n" + "=" * 60)
    print("  汇总对比表")
    print("=" * 60)
    print(f"\n{'方案':<22} {'速度中位误':>10} {'速度<5cm/s':>12} {'行程中位误':>12}")
    print("-" * 60)
    print(f"{'① 直接欧拉积分':<22} {np.median(err1_v):>10.1f} {100*(err1_v<5).mean():>11.1f}% {np.median([abs(t[1]-t[0]) for t in trips1]):>11.1f} cm")
    print(f"{'② KF三阶卡尔曼':<22} {np.median(err2_v):>10.1f} {100*(err2_v<5).mean():>11.1f}% {np.median([abs(t[1]-t[0]) for t in trips2]):>11.1f} cm")
    print(f"{'③ 低通+积分':<22} {np.median(err3_v):>10.1f} {100*(err3_v<5).mean():>11.1f}% {np.median([abs(t[1]-t[0]) for t in trips3]):>11.1f} cm")

    # ---- 可视化 ----
    results = {
        '直接积分': (v1, s1, None),
        'KF三阶': (v2, s2, None),
        '低通+积分': (v3, s3, None),
    }
    plot_comparison(df, results, row_start=0, row_end=300)

    # ---- 保存 ----
    out = pd.DataFrame({
        'utc_time': df['utc_time'],
        'speed_orig': df['speed_cm_s'],
        'dist_orig': df['distance_live_cm'],
        'v_direct': v1, 's_direct': s1,
        'v_kf': v2, 's_kf': s2,
        'v_lowpass': v3, 's_lowpass': s3,
    })
    out.to_csv('/workspace/kalman_comparison_result.csv', index=False, encoding='utf-8')
    print("\n✓ 详细数据已保存: /workspace/kalman_comparison_result.csv")


if __name__ == '__main__':
    main()
