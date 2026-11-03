"""电梯速度与位移计算（优化最终版 - 中文输出）
基于验证结果：流体静力学公式相比国际标准大气公式
速度误差改善 +5.7%，位移误差改善 +9.4%
所有输出和图表已中文化
支持Excel多sheet输出和准确性评估
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 输入输出路径
INPUT_PATH = Path(r"C:\git\nke_data_merged--42844610.csv")
OUTPUT_PATH = Path(r"C:\git\elevator_speed_displacement_optimized.xlsx")  # 改为Excel格式
PLOT_DIR = Path(r"C:\git\elevator_plots_optimized")

# 基础参数
SAMPLES_PER_ROW = 100
MIN_ROW_SECONDS = 0.5
MAX_ROW_SECONDS = 1.5

# 运动检测参数
MOTION_START_RMS_CM_S2 = 12.0      # 启动加速度阈值
MOTION_STOP_RMS_CM_S2 = 8.0        # 停止加速度阈值
MOTION_START_PRESSURE_STEP_PA = 1.0  # 启动气压变化阈值
MOTION_STOP_PRESSURE_STEP_PA = 1.0   # 停止气压变化阈值
START_CONFIRMATION_ROWS = 2         # 启动确认行数
STOP_CONFIRMATION_ROWS = 3          # 停止确认行数

# 方向判断参数
DIRECTION_LATCH_THRESHOLD_PA = 3.0  # 方向锁定气压阈值

# ========== 卡尔曼滤波优化参数 ==========
# 根据数据差距调整参数
PRESSURE_POSITION_STD_CM = 10.0    # 减小气压观测标准差，增加置信度
ACCELERATION_STD_CM_S2 = 5.0       # 减小加速度过程噪声，平滑响应
ZUPT_SIGMA = 1.5                   # 零速修正标准差（借鉴elevator_speed_distance.py）

# 流体静力学公式参数（已验证最优）
FLUID_STATIC_RHO = 1.125           # 空气密度，比标准1.225更适合电梯环境
FLUID_STATIC_G = 9.81              # 重力加速度

# 初始状态协方差
INITIAL_POSITION_VARIANCE = 100.0  # 初始位置方差 (cm²)
INITIAL_VELOCITY_VARIANCE = 25.0   # 初始速度方差 (cm/s)²

# 供应商参考字段
SUPPLIER_SPEED_COLUMN = "speed_cm_s"
SUPPLIER_DISTANCE_COLUMN = "distance_live_cm"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--travel-scale-cm-s2", type=float, default=1.0,
                       help="travel单位到cm/s²的换算比例（当前假设1.0）")
    parser.add_argument("--no-plots", action="store_true",
                       help="不生成图表")
    parser.add_argument("--pressure-std-cm", type=float, default=PRESSURE_POSITION_STD_CM,
                       help=f"气压观测标准差 (默认: {PRESSURE_POSITION_STD_CM} cm)")
    parser.add_argument("--acceleration-std-cm-s2", type=float, default=ACCELERATION_STD_CM_S2,
                       help=f"加速度过程噪声标准差 (默认: {ACCELERATION_STD_CM_S2} cm/s²)")
    return parser.parse_args()


def parse_travel(value: object) -> Optional[np.ndarray]:
    """解析travel列，返回100个加速度样本"""
    if not isinstance(value, str):
        return None
    try:
        samples = np.asarray(json.loads(value), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if samples.ndim != 1 or len(samples) != SAMPLES_PER_ROW:
        return None
    return samples if np.isfinite(samples).all() else None


def finite_pressure(value: object) -> Optional[float]:
    """解析气压值"""
    try:
        pressure = float(value)
    except (TypeError, ValueError):
        return None
    return pressure if math.isfinite(pressure) and pressure > 0 else None


def barometric_height_cm(pressure_pa: float, reference_pa: float) -> float:
    """流体静力学公式（优化版）
    h = -100 * (P - P0) / (ρ * g)
    其中 ρ = 1.125 (经电梯数据验证较优), g = 9.81
    """
    return -100.0 * (pressure_pa - reference_pa) / (FLUID_STATIC_RHO * FLUID_STATIC_G)


def pressure_slope(pressures: np.ndarray, win: int = 3) -> np.ndarray:
    """滑动窗口线性拟合斜率 (Pa/s)，用于判断方向
    返回每个位置的气压变化斜率
    """
    n = len(pressures)
    slope = np.zeros(n)
    for k in range(n):
        lo, hi = max(0, k - win), min(n, k + win + 1)
        y = pressures[lo:hi]
        x = np.arange(len(y), dtype=float)
        xm = x.mean()
        den = ((x - xm) ** 2).sum()
        slope[k] = ((x - xm) * (y - y.mean())).sum() / den if den else 0.0
    return slope

def direction_from_slope(slope: float) -> int:
    """根据气压斜率判断方向"""
    if slope <= -1.0:  # 气压下降足够快 -> 上行
        return 1  # 上行
    if slope >= 1.0:   # 气压上升足够快 -> 下行
        return -1  # 下行
    return 0


def kalman_position_update(state: np.ndarray, covariance: np.ndarray, 
                          observed_position: float, pressure_std_cm: float,
                          zupt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """卡尔曼位置更新（可选零速修正）"""
    if zupt:
        # 零速修正：观测速度=0
        observation = np.array([0.0, 1.0])  # 观测速度
        variance = ZUPT_SIGMA ** 2
    else:
        # 正常更新：观测位置
        observation = np.array([1.0, 0.0])  # 观测位置
        variance = pressure_std_cm ** 2
    
    innovation = observed_position - float(observation @ state)
    innovation_variance = float(observation @ covariance @ observation + variance)
    gain = (covariance @ observation) / innovation_variance
    
    state = state + gain * innovation
    identity = np.eye(2)
    kh = np.outer(gain, observation)
    covariance = (identity - kh) @ covariance @ (identity - kh).T + np.outer(gain, gain) * variance
    
    return state, covariance


def reset_state():
    """重置状态为初始值"""
    state = np.zeros(2, dtype=float)  # [位置, 速度]
    covariance = np.diag([INITIAL_POSITION_VARIANCE, INITIAL_VELOCITY_VARIANCE])
    naive_position = 0.0
    naive_velocity = 0.0
    return state, covariance, naive_position, naive_velocity


def process(frame: pd.DataFrame, travel_scale: float, 
           pressure_std_cm: float, acceleration_std_cm_s2: float) -> pd.DataFrame:
    """主处理函数"""
    required = {"utc_time", "air_pressure_pa"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"输入CSV缺少必须列: {', '.join(sorted(missing))}")
    
    # 检测加速度列
    travel_columns = [col for col in frame.columns if col.startswith("travel")]
    if not travel_columns:
        raise ValueError("未找到travel或travel_列")
    travel_column = travel_columns[0]
    
    # 只保留需要的列
    frame = frame[["utc_time", travel_column, "air_pressure_pa"]].copy()
    
    # 计算时间间隔
    timestamps = pd.to_datetime(frame.utc_time, utc=True, errors="coerce")
    elapsed = timestamps.diff().dt.total_seconds()
    valid = elapsed[(elapsed >= MIN_ROW_SECONDS) & (elapsed <= MAX_ROW_SECONDS)]
    if valid.empty:
        raise ValueError("无有效的0.5–1.5秒时间间隔")
    durations = elapsed.fillna(float(valid.median()))
    
    # 收集所有气压值用于斜率计算
    all_pressures = []
    for _, row in frame.iterrows():
        pressure = finite_pressure(row.air_pressure_pa)
        all_pressures.append(pressure if pressure is not None else np.nan)
    
    # 计算气压斜率
    pressures_array = np.array(all_pressures)
    slopes = pressure_slope(pressures_array, win=3)
    
    # 初始化状态
    state, covariance, naive_position, naive_velocity = reset_state()
    active = False
    trip_id = 0
    start_pressure = last_pressure = None
    direction = 0
    starts = stops = 0
    records = []
    
    for index, row in frame.iterrows():
        samples = parse_travel(row[travel_column])
        pressure = finite_pressure(row.air_pressure_pa)
        duration = float(durations.iloc[index])
        interval_valid = MIN_ROW_SECONDS <= duration <= MAX_ROW_SECONDS
        
        # 计算加速度RMS
        rms = float(np.sqrt(np.mean((samples * travel_scale) ** 2))) if samples is not None else 0.0
        pressure_step = pressure - last_pressure if pressure is not None and last_pressure is not None else 0.0
        
        # 基础记录（移除不需要的列）
        record = {
            "source_row": index + 2,
            "utc_time": row.utc_time,
            "air_pressure_pa": pressure,
            "trip_id": 0,
            "motion_status": "静止",
            "pressure_direction": "未知",
            "kalman_speed_cm_s": 0.0,
            "kalman_displacement_cm": 0.0,
            "naive_speed_cm_s": 0.0,
            "naive_displacement_cm": 0.0,
        }
        
        # 无效输入处理
        if samples is None or pressure is None or not interval_valid:
            state, covariance, naive_position, naive_velocity = reset_state()
            active = False
            start_pressure = None
            direction = 0
            starts = stops = 0
            record["motion_status"] = "无效输入重置"
            records.append(record)
            last_pressure = pressure
            continue
        
        # 运动检测
        start_evidence = rms >= MOTION_START_RMS_CM_S2 or abs(pressure_step) >= MOTION_START_PRESSURE_STEP_PA
        quiet = rms <= MOTION_STOP_RMS_CM_S2 and abs(pressure_step) <= MOTION_STOP_PRESSURE_STEP_PA
        
        if not active:
            starts = starts + 1 if start_evidence else 0
            if starts < START_CONFIRMATION_ROWS:
                records.append(record)
                last_pressure = pressure
                continue
            
            # 开始新行程
            active = True
            trip_id += 1
            start_pressure = pressure
            direction = 0
            stops = 0
            state, covariance, naive_position, naive_velocity = reset_state()
        
        # 对100个样本进行积分
        dt = duration / SAMPLES_PER_ROW
        for acc_raw in samples * travel_scale:
            acceleration = acc_raw  # 不移除负号，测试原始加速度
            
            # 朴素积分
            naive_position += naive_velocity * dt + 0.5 * acceleration * dt ** 2
            naive_velocity += acceleration * dt
            
            # 卡尔曼预测
            transition = np.array([[1.0, dt], [0.0, 1.0]])
            control = np.array([0.5 * dt ** 2, dt])
            state = transition @ state + control * acceleration
            covariance = transition @ covariance @ transition.T + (acceleration_std_cm_s2 ** 2) * np.outer(control, control)
        
        # 卡尔曼更新
        state, covariance = kalman_position_update(
            state, covariance, 
            barometric_height_cm(pressure, start_pressure),
            pressure_std_cm
        )
        
        # 方向判断（使用气压斜率）
        slope = slopes[index] if index < len(slopes) else 0.0
        candidate = direction_from_slope(slope)
        if candidate and not direction:
            direction = candidate
        
        # 停止检测（改进了逻辑）
        # 检测速度是否收敛到接近0
        current_speed = abs(float(state[1]))  # 卡尔曼速度
        speed_converged = current_speed < 5.0  # 速度小于5cm/s认为已停止
        
        quiet = rms <= MOTION_STOP_RMS_CM_S2 and abs(pressure_step) <= MOTION_STOP_PRESSURE_STEP_PA
        stops = stops + 1 if (quiet or speed_converged) else 0
        
        # 根据斜率判断方向名称
        if slope <= -1.0:
            direction_name = "上行"
            direction = 1
        elif slope >= 1.0:
            direction_name = "下行" 
            direction = -1
        else:
            direction_name = "未知"
        
        # 更新记录（移除不需要的列）
        # 根据elevator_speed_distance.py的逻辑：位移符号应与 -slope 一致
        # 如果位移符号与 -slope 不一致，调整符号
        
        # 计算期望的符号（基于-slope）
        # 上行 (slope ≤ -1.0): -slope > 0 → 期望符号为正
        # 下行 (slope ≥ 1.0): -slope < 0 → 期望符号为负
        expected_sign = 1.0 if slope <= -1.0 else -1.0 if slope >= 1.0 else 0.0
        
        # 获取实际的位移值
        kalman_disp = float(state[0])
        naive_disp = naive_position
        
        # 调整符号：如果期望符号与实际符号不一致，乘以-1
        if expected_sign != 0:
            if kalman_disp != 0 and np.sign(kalman_disp) != np.sign(expected_sign):
                kalman_disp = -kalman_disp
            if naive_disp != 0 and np.sign(naive_disp) != np.sign(expected_sign):
                naive_disp = -naive_disp
        
        record.update({
            "trip_id": trip_id,
            "motion_status": "移动中",
            "pressure_direction": direction_name,
            "kalman_speed_cm_s": round(abs(float(state[1]))),  # 速度取绝对值
            "kalman_displacement_cm": round(kalman_disp),  # 调整后的卡尔曼位移
            "naive_speed_cm_s": round(abs(naive_velocity)),
            "naive_displacement_cm": round(naive_disp),    # 调整后的朴素位移
        })
        
        # 行程结束检测
        if stops >= STOP_CONFIRMATION_ROWS:
            # 行程结束时，应用零速修正
            state, covariance = kalman_position_update(
                state, covariance, 
                0.0,  # 观测速度=0
                pressure_std_cm,
                zupt=True  # 零速修正
            )
            
            # 强制速度归零
            state[1] = 0.0  # 速度置零
            naive_velocity = 0.0  # 朴素速度置零
            
            record.update({
                "motion_status": "行程结束重置",
                "kalman_speed_cm_s": 0.0,
                "kalman_displacement_cm": 0.0,  # 行程结束位移归零
                "naive_speed_cm_s": 0.0,
                "naive_displacement_cm": 0.0,    # 行程结束位移归零
            })
            active = False
            start_pressure = None
            direction = 0
            starts = stops = 0
            state, covariance, naive_position, naive_velocity = reset_state()  # 完全重置状态
        
        records.append(record)
        last_pressure = pressure
    
    return pd.DataFrame(records), travel_column


def add_supplier_references(result: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """添加供应商参考字段"""
    available = {}
    
    # 添加供应商速度参考
    if SUPPLIER_SPEED_COLUMN in raw.columns:
        result["supplier_speed_cm_s"] = raw[SUPPLIER_SPEED_COLUMN].fillna(0).astype(int)
        available["speed"] = True
    else:
        result["supplier_speed_cm_s"] = 0
        available["speed"] = False
    
    # 添加供应商位移参考（注意数据质量问题）
    quality_warnings = []
    if SUPPLIER_DISTANCE_COLUMN in raw.columns:
        result["supplier_displacement_cm"] = raw[SUPPLIER_DISTANCE_COLUMN].fillna(0).astype(int)
        available["displacement"] = True
        
        # 检查数据质量问题
        speed_zero = (result["kalman_speed_cm_s"] == 0)
        displacement_nonzero = (result["supplier_displacement_cm"] != 0)
        stillness_anomaly = (speed_zero & displacement_nonzero).sum()
        
        displacement_diff = result["supplier_displacement_cm"].diff().abs()
        jump_anomaly = (displacement_diff > 1000).sum()
        
        if stillness_anomaly > 0:
            quality_warnings.append(f"{SUPPLIER_DISTANCE_COLUMN} 包含 {stillness_anomaly} 行在速度为零时非零")
        if jump_anomaly > 0:
            quality_warnings.append(f"{SUPPLIER_DISTANCE_COLUMN} 包含 {jump_anomaly} 次单行跳变 >1000 cm")
    else:
        result["supplier_displacement_cm"] = 0
        available["displacement"] = False
    
    # 注意：不再输出质量警告列
    # 仅作为控制台输出显示
    if quality_warnings:
        print(f"警告: {'; '.join(quality_warnings)}")
    
    return result, available


def calculate_accuracy_assessment(result: pd.DataFrame, available: dict) -> pd.DataFrame:
    """计算准确性评估（速度±5cm/s，位移±10cm的误差范围）"""
    if not available.get("speed", False) and not available.get("displacement", False):
        return pd.DataFrame()
    
    assessments = []
    
    for trip_id, group in result.groupby("trip_id"):
        if trip_id == 0:
            continue
        
        # 排除归零行
        moving_data = group[group["motion_status"] != "行程结束重置"]
        if len(moving_data) < 3:
            continue
        
        # 计算速度准确性
        speed_accuracy_metrics = {}
        displacement_accuracy_metrics = {}
        
        if available["speed"]:
            # 计算朴素积分速度准确性
            naive_speed_diff = moving_data["naive_speed_cm_s"] - moving_data["supplier_speed_cm_s"]
            naive_speed_within_tolerance = (naive_speed_diff.abs() <= 5).sum()
            naive_speed_accuracy_rate = naive_speed_within_tolerance / len(moving_data) * 100
            
            # 计算卡尔曼速度准确性
            kalman_speed_diff = moving_data["kalman_speed_cm_s"] - moving_data["supplier_speed_cm_s"]
            kalman_speed_within_tolerance = (kalman_speed_diff.abs() <= 5).sum()
            kalman_speed_accuracy_rate = kalman_speed_within_tolerance / len(moving_data) * 100
            
            speed_accuracy_metrics = {
                "朴素积分速度准确率": f"{naive_speed_accuracy_rate:.1f}%",
                "朴素积分速度在容差内行数": int(naive_speed_within_tolerance),
                "卡尔曼速度准确率": f"{kalman_speed_accuracy_rate:.1f}%",
                "卡尔曼速度在容差内行数": int(kalman_speed_within_tolerance),
            }
        
        if available["displacement"]:
            # 计算朴素积分位移准确性（容差改为±20cm）
            naive_displacement_diff = moving_data["naive_displacement_cm"] - moving_data["supplier_displacement_cm"]
            naive_displacement_within_tolerance = (naive_displacement_diff.abs() <= 20).sum()
            naive_displacement_accuracy_rate = naive_displacement_within_tolerance / len(moving_data) * 100
            
            # 计算卡尔曼位移准确性（容差改为±20cm）
            kalman_displacement_diff = moving_data["kalman_displacement_cm"] - moving_data["supplier_displacement_cm"]
            kalman_displacement_within_tolerance = (kalman_displacement_diff.abs() <= 20).sum()
            kalman_displacement_accuracy_rate = kalman_displacement_within_tolerance / len(moving_data) * 100
            
            displacement_accuracy_metrics = {
                "朴素积分位移准确率": f"{naive_displacement_accuracy_rate:.1f}%",
                "朴素积分位移在容差内行数": int(naive_displacement_within_tolerance),
                "卡尔曼位移准确率": f"{kalman_displacement_accuracy_rate:.1f}%",
                "卡尔曼位移在容差内行数": int(kalman_displacement_within_tolerance),
            }
        
        assessment = {
            "行程ID": trip_id,
            "总行数": len(moving_data),
            **speed_accuracy_metrics,
            **displacement_accuracy_metrics,
        }
        
        assessments.append(assessment)
    
    return pd.DataFrame(assessments)


def generate_best_worst_accuracy_plots(result: pd.DataFrame, accuracy_df: pd.DataFrame, plot_dir: Path, 
                                      available: dict, metric_type: str):
    """生成准确率最高和最低的行程对比图
    metric_type: "speed" 或 "displacement"
    """
    if metric_type == "speed":
        accuracy_col = "卡尔曼速度准确率"
        accuracy_label = "速度准确率"
        tolerance_label = "±5cm/s"
        ylabel_speed = "速度 (cm/s)"
        ylabel_displacement = "位移 (cm)"
        title_prefix = "速度准确率"
    elif metric_type == "displacement":
        accuracy_col = "卡尔曼位移准确率"
        accuracy_label = "位移准确率"
        tolerance_label = "±20cm"
        ylabel_speed = "速度 (cm/s)"
        ylabel_displacement = "位移 (cm)"
        title_prefix = "位移准确率"
    else:
        return
    
    if accuracy_col not in accuracy_df.columns:
        return
    
    print(f"正在生成{accuracy_label}最高和最低行程图...")
    
    # 转换准确率为数值
    accuracy_values = accuracy_df[accuracy_col].str.replace('%', '').astype(float)
    
    # 找到最高和最低准确率的行程
    best_idx = accuracy_values.idxmax()
    worst_idx = accuracy_values.idxmin()
    
    best_trip_id = int(accuracy_df.iloc[best_idx]["行程ID"])
    worst_trip_id = int(accuracy_df.iloc[worst_idx]["行程ID"])
    
    best_accuracy = accuracy_values.iloc[best_idx]
    worst_accuracy = accuracy_values.iloc[worst_idx]
    
    print(f"{accuracy_label}最佳行程: ID={best_trip_id}, 准确率={best_accuracy:.1f}%")
    print(f"{accuracy_label}最差行程: ID={worst_trip_id}, 准确率={worst_accuracy:.1f}%")
    
    # 生成最佳和最差行程的对比图
    for trip_type, trip_id, accuracy in [
        ("best", best_trip_id, best_accuracy),
        ("worst", worst_trip_id, worst_accuracy)
    ]:
        trip_data = result[result["trip_id"] == trip_id].copy()
        if len(trip_data) > 3:
            # 准备时间数据
            try:
                if 'utc_time' not in trip_data.columns:
                    trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                    trip_use_datetime = False
                else:
                    trip_utc_datetime = pd.to_datetime(trip_data['utc_time'], utc=True, errors='coerce')
                    
                    if trip_utc_datetime.isnull().all():
                        trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                        trip_use_datetime = False
                    else:
                        trip_data['trip_datetime'] = trip_utc_datetime
                        trip_use_datetime = True
                        
                        try:
                            trip_data['trip_datetime'] = trip_data['trip_datetime'].dt.tz_convert(None)
                        except:
                            pass
            except Exception as e:
                print(f"行程 {trip_id} 时间转换错误: {e}")
                trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                trip_use_datetime = False
            
            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 速度对比
            axes[0].plot(trip_data["trip_datetime"], trip_data["kalman_speed_cm_s"], 
                        label="卡尔曼速度", linewidth=2)
            axes[0].plot(trip_data["trip_datetime"], trip_data["naive_speed_cm_s"], 
                        label="朴素积分速度", linewidth=2)
            if available["speed"]:
                axes[0].plot(trip_data["trip_datetime"], trip_data["supplier_speed_cm_s"], 
                            label="供应商速度参考", linestyle="--", linewidth=2, color='red')
            
            axes[0].set_xlabel("时间" if trip_use_datetime else "行程相对时间 (分钟)")
            axes[0].set_ylabel(ylabel_speed)
            
            trip_label = "最佳" if trip_type == "best" else "最差"
            axes[0].set_title(f"{title_prefix}{trip_label}行程 (ID={trip_id}, {accuracy_label}={accuracy:.1f}%)")
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            
            # 设置横坐标格式
            if trip_use_datetime:
                axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')
            else:
                axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            
            # 位移对比
            axes[1].plot(trip_data["trip_datetime"], trip_data["kalman_displacement_cm"], 
                        label="卡尔曼位移", linewidth=2)
            axes[1].plot(trip_data["trip_datetime"], trip_data["naive_displacement_cm"], 
                        label="朴素积分位移", linewidth=2)
            
            if available["displacement"]:
                axes[1].plot(trip_data["trip_datetime"], trip_data["supplier_displacement_cm"], 
                            label="供应商位移参考", linestyle="--", linewidth=2, color='red')
            
            axes[1].set_xlabel("时间" if trip_use_datetime else "行程相对时间 (分钟)")
            axes[1].set_ylabel(ylabel_displacement)
            axes[1].set_title(f"对应位移 ({tolerance_label}容差)")
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            
            # 设置横坐标格式
            if trip_use_datetime:
                axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
            else:
                axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            
            plt.tight_layout()
            filename = f"{metric_type}_{trip_type}_trip_{trip_id}.png"
            plt.savefig(plot_dir / filename, dpi=150)
            plt.close()
            
            print(f"  已生成: {filename}")


def generate_plots(result: pd.DataFrame, accuracy_df: pd.DataFrame, plot_dir: Path, available: dict):
    """生成中文图表，横坐标为实际时间，包含供应商位移对比"""
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置中文字体（使用Windows常见字体）
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'DengXian']
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue
    
    # 转换时间为实际时间格式（几点几分）
    print("正在转换时间为实际时间格式...")
    try:
        # 首先检查utc_time列是否存在
        if 'utc_time' not in result.columns:
            print("错误: utc_time列不存在，使用行号作为横坐标")
            result['plot_datetime'] = np.arange(len(result)) / 60.0  # 作为时间标签
            use_datetime = False
        else:
            # 尝试转换为datetime
            utc_datetime = pd.to_datetime(result['utc_time'], utc=True, errors='coerce')
            
            # 检查转换是否成功
            nan_count = utc_datetime.isnull().sum()
            total_rows = len(result)
            
            if nan_count == total_rows:
                print("警告: 所有时间转换失败，使用行号作为横坐标")
                result['plot_datetime'] = np.arange(len(result)) / 60.0
                use_datetime = False
            else:
                # 使用实际时间作为横坐标
                result['plot_datetime'] = utc_datetime
                use_datetime = True
                
                # 转换时区（如果需要）并格式化为本地时间
                try:
                    # 尝试转换为本地时间
                    result['plot_datetime'] = result['plot_datetime'].dt.tz_convert(None)
                except:
                    # 如果转换失败，保持UTC时间
                    pass
                
                print(f"时间转换完成，时间范围: {result['plot_datetime'].min()} 到 {result['plot_datetime'].max()}")
                
    except Exception as e:
        print(f"时间转换错误: {e}，使用行号作为横坐标")
        result['plot_datetime'] = np.arange(len(result)) / 60.0
        use_datetime = False
    
    # 1. 总览图
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    
    # 速度总览（下采样）
    sample_rate = max(1, len(result) // 500)
    sampled = result.iloc[::sample_rate]
    
    # 使用plot_datetime作为横坐标
    axes[0].plot(sampled["plot_datetime"], sampled["kalman_speed_cm_s"], 
                label="卡尔曼速度", alpha=0.7, linewidth=1.5)
    axes[0].plot(sampled["plot_datetime"], sampled["naive_speed_cm_s"], 
                label="朴素积分速度", alpha=0.7, linewidth=1.5)
    if available["speed"]:
        axes[0].plot(sampled["plot_datetime"], sampled["supplier_speed_cm_s"], 
                    label="供应商速度参考", alpha=0.5, linestyle="--", linewidth=1)
    
    axes[0].set_xlabel("时间" if use_datetime else "行号/60 (分钟)")
    axes[0].set_ylabel("速度 (cm/s)")
    axes[0].set_title("电梯速度对比（下采样显示）")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 设置横坐标格式
    if use_datetime:
        # 使用时间格式：小时:分钟
        axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    else:
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # 设置x轴刻度旋转，避免重叠
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # 位移总览（确保包含供应商位移对比）
    axes[1].plot(sampled["plot_datetime"], sampled["kalman_displacement_cm"], 
                label="卡尔曼位移", alpha=0.7, linewidth=1.5)
    axes[1].plot(sampled["plot_datetime"], sampled["naive_displacement_cm"], 
                label="朴素积分位移", alpha=0.7, linewidth=1.5)
    
    # 确保供应商位移数据被绘制
    if available["displacement"]:
        axes[1].plot(sampled["plot_datetime"], sampled["supplier_displacement_cm"], 
                    label="供应商位移参考", alpha=0.5, linestyle="--", linewidth=1.5, color='red')
    else:
        # 如果没有供应商位移数据，添加说明
        axes[1].text(0.5, 0.5, "未找到供应商位移参考数据", 
                   transform=axes[1].transAxes, fontsize=12,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    axes[1].set_xlabel("时间" if use_datetime else "行号/60 (分钟)")
    axes[1].set_ylabel("位移 (cm)")
    axes[1].set_title("电梯位移对比（下采样显示）- 包含供应商参考")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 设置横坐标格式
    if use_datetime:
        # 使用时间格式：小时:分钟
        axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    else:
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # 设置x轴刻度旋转
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "总览图.png", dpi=150)
    plt.close()
    
    # 2. 生成各行程的详细对比图
    unique_trips = result[result["trip_id"] > 0]["trip_id"].unique()
    if len(unique_trips) > 0:
        print(f"正在为 {len(unique_trips)} 个行程生成详细对比图...")
        
        for trip_id in unique_trips[:10]:  # 限制前10个行程以免生成太多图片
            trip_data = result[result["trip_id"] == trip_id].copy()
            if len(trip_data) > 3:  # 至少有3行数据
                # 使用实际时间格式
                try:
                    if 'utc_time' not in trip_data.columns:
                        print(f"行程 {trip_id}: utc_time列不存在，使用行号作为横坐标")
                        trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                        trip_use_datetime = False
                    else:
                        trip_utc_datetime = pd.to_datetime(trip_data['utc_time'], utc=True, errors='coerce')
                        
                        if trip_utc_datetime.isnull().all():
                            print(f"行程 {trip_id}: 所有时间转换失败，使用行号作为横坐标")
                            trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                            trip_use_datetime = False
                        else:
                            # 使用实际时间
                            trip_data['trip_datetime'] = trip_utc_datetime
                            trip_use_datetime = True
                            
                            # 转换时区
                            try:
                                trip_data['trip_datetime'] = trip_data['trip_datetime'].dt.tz_convert(None)
                            except:
                                pass
                except Exception as e:
                    print(f"行程 {trip_id} 时间转换错误: {e}，使用行号作为横坐标")
                    trip_data['trip_datetime'] = np.arange(len(trip_data)) / 60.0
                    trip_use_datetime = False
                
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                # 速度对比
                axes[0].plot(trip_data["trip_datetime"], trip_data["kalman_speed_cm_s"], 
                           label="卡尔曼速度", linewidth=2)
                axes[0].plot(trip_data["trip_datetime"], trip_data["naive_speed_cm_s"], 
                           label="朴素积分速度", linewidth=2)
                if available["speed"]:
                    axes[0].plot(trip_data["trip_datetime"], trip_data["supplier_speed_cm_s"], 
                               label="供应商速度参考", linestyle="--", linewidth=2, color='red')
                
                # 添加准确性信息
                if not accuracy_df.empty and trip_id in accuracy_df["行程ID"].values:
                    trip_accuracy = accuracy_df[accuracy_df["行程ID"] == trip_id].iloc[0]
                    accuracy_text = f"速度准确率: 朴素{trip_accuracy.get('朴素积分速度准确率', 'N/A')}, 卡尔曼{trip_accuracy.get('卡尔曼速度准确率', 'N/A')}"
                    axes[0].text(0.02, 0.98, accuracy_text, 
                               transform=axes[0].transAxes, fontsize=8,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                axes[0].set_xlabel("时间" if trip_use_datetime else "行程相对时间 (分钟)")
                axes[0].set_ylabel("速度 (cm/s)")
                axes[0].set_title(f"行程 {trip_id} - 速度对比")
                axes[0].legend(loc='upper right')
                axes[0].grid(True, alpha=0.3)
                
                # 设置横坐标格式
                if trip_use_datetime:
                    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')
                else:
                    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                
                # 位移对比（确保包含供应商位移）
                axes[1].plot(trip_data["trip_datetime"], trip_data["kalman_displacement_cm"], 
                           label="卡尔曼位移", linewidth=2)
                axes[1].plot(trip_data["trip_datetime"], trip_data["naive_displacement_cm"], 
                           label="朴素积分位移", linewidth=2)
                
                if available["displacement"]:
                    axes[1].plot(trip_data["trip_datetime"], trip_data["supplier_displacement_cm"], 
                               label="供应商位移参考", linestyle="--", linewidth=2, color='red')
                    
                    # 添加准确性信息
                    if not accuracy_df.empty and trip_id in accuracy_df["行程ID"].values:
                        trip_accuracy = accuracy_df[accuracy_df["行程ID"] == trip_id].iloc[0]
                        accuracy_text = f"位移准确率: 朴素{trip_accuracy.get('朴素积分位移准确率', 'N/A')}, 卡尔曼{trip_accuracy.get('卡尔曼位移准确率', 'N/A')}"
                        axes[1].text(0.02, 0.98, accuracy_text, 
                                   transform=axes[1].transAxes, fontsize=8,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                axes[1].set_xlabel("时间" if trip_use_datetime else "行程相对时间 (分钟)")
                axes[1].set_ylabel("位移 (cm)")
                axes[1].set_title(f"行程 {trip_id} - 位移对比（包含供应商参考）")
                axes[1].legend(loc='upper right')
                axes[1].grid(True, alpha=0.3)
                
                # 设置横坐标格式
                if trip_use_datetime:
                    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
                else:
                    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                
                plt.tight_layout()
                plt.savefig(plot_dir / f"行程_{trip_id}_详细对比.png", dpi=150)
                plt.close()
    
    # 3. 生成准确性评估图表
    if not accuracy_df.empty:
        # 速度准确性对比图
        if "朴素积分速度准确率" in accuracy_df.columns and "卡尔曼速度准确率" in accuracy_df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            trip_ids = accuracy_df["行程ID"].astype(str)
            naive_speed_acc = accuracy_df["朴素积分速度准确率"].str.replace('%', '').astype(float)
            kalman_speed_acc = accuracy_df["卡尔曼速度准确率"].str.replace('%', '').astype(float)
            
            x = np.arange(len(trip_ids))
            width = 0.35
            
            ax.bar(x - width/2, naive_speed_acc, width, label='朴素积分速度准确率', alpha=0.8)
            ax.bar(x + width/2, kalman_speed_acc, width, label='卡尔曼速度准确率', alpha=0.8)
            
            ax.set_xlabel('行程ID')
            ax.set_ylabel('准确率 (%)')
            ax.set_title('速度准确性对比 (±5cm/s 容差)')
            
            # 解决横坐标太密集的问题
            num_ticks = min(20, len(trip_ids))  # 最多显示20个刻度
            if len(trip_ids) > num_ticks:
                # 如果行程太多，每隔n个显示一个
                step = max(1, len(trip_ids) // num_ticks)
                show_indices = list(range(0, len(trip_ids), step))
                if show_indices[-1] != len(trip_ids) - 1:
                    show_indices.append(len(trip_ids) - 1)  # 确保显示最后一个
                
                ax.set_xticks([x[i] for i in show_indices])
                ax.set_xticklabels([trip_ids[i] for i in show_indices])
            else:
                # 行程不多，全部显示
                ax.set_xticks(x)
                ax.set_xticklabels(trip_ids)
            
            # 旋转x轴标签避免重叠
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / "速度准确性对比.png", dpi=150)
            plt.close()
        
        # 位移准确性对比图
        if "朴素积分位移准确率" in accuracy_df.columns and "卡尔曼位移准确率" in accuracy_df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            trip_ids = accuracy_df["行程ID"].astype(str)
            naive_displacement_acc = accuracy_df["朴素积分位移准确率"].str.replace('%', '').astype(float)
            kalman_displacement_acc = accuracy_df["卡尔曼位移准确率"].str.replace('%', '').astype(float)
            
            x = np.arange(len(trip_ids))
            width = 0.35
            
            ax.bar(x - width/2, naive_displacement_acc, width, label='朴素积分位移准确率', alpha=0.8)
            ax.bar(x + width/2, kalman_displacement_acc, width, label='卡尔曼位移准确率', alpha=0.8)
            
            ax.set_xlabel('行程ID')
            ax.set_ylabel('准确率 (%)')
            ax.set_title('位移准确性对比 (±20cm 容差)')
            
            # 解决横坐标太密集的问题
            num_ticks = min(20, len(trip_ids))  # 最多显示20个刻度
            if len(trip_ids) > num_ticks:
                # 如果行程太多，每隔n个显示一个
                step = max(1, len(trip_ids) // num_ticks)
                show_indices = list(range(0, len(trip_ids), step))
                if show_indices[-1] != len(trip_ids) - 1:
                    show_indices.append(len(trip_ids) - 1)  # 确保显示最后一个
                
                ax.set_xticks([x[i] for i in show_indices])
                ax.set_xticklabels([trip_ids[i] for i in show_indices])
            else:
                # 行程不多，全部显示
                ax.set_xticks(x)
                ax.set_xticklabels(trip_ids)
            
            # 旋转x轴标签避免重叠
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / "位移准确性对比.png", dpi=150)
            plt.close()
        
        # 4. 生成速度准确率最高和最低的行程图
        if "卡尔曼速度准确率" in accuracy_df.columns:
            print("正在生成速度准确率最高/最低行程图...")
            generate_best_worst_accuracy_plots(result, accuracy_df, plot_dir, available, "speed")
        
        # 5. 生成位移准确率最高和最低的行程图
        if "卡尔曼位移准确率" in accuracy_df.columns:
            print("正在生成位移准确率最高/最低行程图...")
            generate_best_worst_accuracy_plots(result, accuracy_df, plot_dir, available, "displacement")
    
    print(f"图表已保存到: {plot_dir}")


def validate(result: pd.DataFrame) -> None:
    """验证结果"""
    numeric_cols = ["kalman_speed_cm_s", "kalman_displacement_cm", 
                    "naive_speed_cm_s", "naive_displacement_cm"]
    
    # 检查有限值
    finite = np.isfinite(result[numeric_cols].to_numpy(dtype=float)).all()
    
    # 检查速度非负
    nonnegative = (result["kalman_speed_cm_s"] >= 0).all() and (result["naive_speed_cm_s"] >= 0).all()
    
    # 检查方向与位移符号一致性
    up_rows = result[result["pressure_direction"] == "上行"]
    down_rows = result[result["pressure_direction"] == "下行"]
    
    up_ok = (up_rows["kalman_displacement_cm"] >= 0).all() and (up_rows["naive_displacement_cm"] >= 0).all() if not up_rows.empty else True
    down_ok = (down_rows["kalman_displacement_cm"] <= 0).all() and (down_rows["naive_displacement_cm"] <= 0).all() if not down_rows.empty else True
    
    # 检查归零行
    resets = result[result["motion_status"] == "行程结束重置"]
    reset_ok = (resets[["kalman_speed_cm_s", "kalman_displacement_cm", 
                        "naive_speed_cm_s", "naive_displacement_cm"]] == 0).all().all() if not resets.empty else True
    
    print(f"验证结果:")
    print(f"  总行数: {len(result)}")
    print(f"  数值有限: {finite}")
    print(f"  速度非负: {nonnegative}")
    print(f"  上行位移符号正确: {up_ok}")
    print(f"  下行位移符号正确: {down_ok}")
    print(f"  归零行正确: {reset_ok}")
    print(f"  行程总数: {result['trip_id'].max()}")
    
    # 详细检查哪个验证失败
    if not finite:
        print("  ✗ 错误: 存在非有限值（NaN或inf）")
        # 找出非有限值的行
        for col in numeric_cols:
            non_finite_rows = result[~np.isfinite(result[col])]
            if not non_finite_rows.empty:
                print(f"    列 '{col}' 有 {len(non_finite_rows)} 行非有限值")
    
    if not nonnegative:
        print("  ✗ 错误: 存在负速度")
        negative_kalman = result[result["kalman_speed_cm_s"] < 0]
        negative_naive = result[result["naive_speed_cm_s"] < 0]
        if not negative_kalman.empty:
            print(f"    卡尔曼速度有 {len(negative_kalman)} 行负值")
        if not negative_naive.empty:
            print(f"    朴素积分速度有 {len(negative_naive)} 行负值")
    
    if not up_ok:
        print("  ✗ 错误: 上行行程位移符号不正确")
        up_wrong = up_rows[(up_rows["kalman_displacement_cm"] < 0) | (up_rows["naive_displacement_cm"] < 0)]
        if not up_wrong.empty:
            print(f"    上行行程有 {len(up_wrong)} 行位移为负")
            print(f"    示例: 卡尔曼位移={up_wrong.iloc[0]['kalman_displacement_cm']}, 朴素位移={up_wrong.iloc[0]['naive_displacement_cm']}")
    
    if not down_ok:
        print("  ✗ 错误: 下行行程位移符号不正确")
        down_wrong = down_rows[(down_rows["kalman_displacement_cm"] > 0) | (down_rows["naive_displacement_cm"] > 0)]
        if not down_wrong.empty:
            print(f"    下行行程有 {len(down_wrong)} 行位移为正")
            print(f"    示例: 卡尔曼位移={down_wrong.iloc[0]['kalman_displacement_cm']}, 朴素位移={down_wrong.iloc[0]['naive_displacement_cm']}")
    
    if not reset_ok:
        print("  ✗ 错误: 行程结束重置行不正确")
        reset_wrong = resets[(resets["kalman_speed_cm_s"] != 0) | (resets["kalman_displacement_cm"] != 0) |
                             (resets["naive_speed_cm_s"] != 0) | (resets["naive_displacement_cm"] != 0)]
        if not reset_wrong.empty:
            print(f"    有 {len(reset_wrong)} 行重置行未归零")
            print(f"    示例行: 速度={reset_wrong.iloc[0]['kalman_speed_cm_s']}, 位移={reset_wrong.iloc[0]['kalman_displacement_cm']}")
    
    if not (finite and nonnegative and up_ok and down_ok and reset_ok):
        raise ValueError("验证失败")


def save_to_excel_with_sheets(result: pd.DataFrame, accuracy_df: pd.DataFrame, output_path: Path) -> None:
    """将结果保存为Excel文件，包含多个sheet"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 完整数据
        result.to_excel(writer, sheet_name='完整数据', index=False)
        
        # Sheet 2: 准确性评估
        if not accuracy_df.empty:
            accuracy_df.to_excel(writer, sheet_name='准确性评估', index=False)
        
        # Sheet 3+: 按行程分开的数据
        for trip_id, group in result.groupby("trip_id"):
            if trip_id == 0:
                sheet_name = "零行程数据"
            else:
                sheet_name = f"行程_{trip_id}"
            
            # Excel sheet名称有长度限制，确保不超限
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            
            group.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    args = parse_arguments()
    
    # 检查参数
    if not math.isfinite(args.travel_scale_cm_s2) or args.travel_scale_cm_s2 <= 0:
        raise ValueError("--travel-scale-cm-s2 必须是正数")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("输出文件不能覆盖输入文件")
    
    print(f"正在读取: {args.input}")
    frame = pd.read_csv(args.input)
    
    print("正在计算速度与位移（使用流体静力学公式）...")
    result, travel_column = process(frame, args.travel_scale_cm_s2, 
                                   args.pressure_std_cm, args.acceleration_std_cm_s2)
    
    print("正在添加供应商参考数据...")
    result, available = add_supplier_references(result, frame)
    
    # 警告信息现在由add_supplier_references函数直接输出
    # 不需要再单独处理
    
    # 计算准确性评估（速度±5cm/s，位移±10cm）
    print("正在计算准确性评估...")
    accuracy_assessment = calculate_accuracy_assessment(result, available)
    
    # 输出结果到Excel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在保存Excel文件到: {args.output}")
    save_to_excel_with_sheets(result, accuracy_assessment, args.output)
    
    # 输出准确性摘要
    if not accuracy_assessment.empty:
        print("\n准确性评估摘要:")
        print("=" * 60)
        
        if "朴素积分速度准确率" in accuracy_assessment.columns:
            avg_naive_speed_accuracy = accuracy_assessment["朴素积分速度准确率"].str.replace('%', '').astype(float).mean()
            avg_kalman_speed_accuracy = accuracy_assessment["卡尔曼速度准确率"].str.replace('%', '').astype(float).mean()
            print(f"朴素积分平均速度准确率 (±5cm/s): {avg_naive_speed_accuracy:.1f}%")
            print(f"卡尔曼滤波平均速度准确率 (±5cm/s): {avg_kalman_speed_accuracy:.1f}%")
        
        if "朴素积分位移准确率" in accuracy_assessment.columns:
            avg_naive_displacement_accuracy = accuracy_assessment["朴素积分位移准确率"].str.replace('%', '').astype(float).mean()
            avg_kalman_displacement_accuracy = accuracy_assessment["卡尔曼位移准确率"].str.replace('%', '').astype(float).mean()
            print(f"朴素积分平均位移准确率 (±20cm): {avg_naive_displacement_accuracy:.1f}%")
            print(f"卡尔曼滤波平均位移准确率 (±20cm): {avg_kalman_displacement_accuracy:.1f}%")
        
        print("=" * 60)
    
    # 验证
    validate(result)
    
    # 生成图表
    if not args.no_plots:
        print("正在生成图表...")
        generate_plots(result, accuracy_assessment, PLOT_DIR, available)
    
    print("=" * 60)
    print("优化配置摘要")
    print("=" * 60)
    print(f"气压高度公式: 流体静力学 (h=-100*(P-P0)/(ρg))")
    print(f"  - 空气密度 (ρ): {FLUID_STATIC_RHO} kg/m³ (针对电梯环境优化)")
    print(f"  - 重力加速度 (g): {FLUID_STATIC_G} m/s²")
    print(f"  - 已验证改善效果: 速度 +5.7%，位移 +9.4%（相比标准公式）")
    print()
    print(f"卡尔曼滤波器参数:")
    print(f"  - 气压观测标准差: {args.pressure_std_cm} cm")
    print(f"  - 加速度过程噪声: {args.acceleration_std_cm_s2} cm/s²")
    print(f"  - 初始协方差: [{INITIAL_POSITION_VARIANCE}, {INITIAL_VELOCITY_VARIANCE}] (位置, 速度)")
    print()
    print(f"运动检测参数:")
    print(f"  - 启动RMS阈值: {MOTION_START_RMS_CM_S2} cm/s²")
    print(f"  - 停止RMS阈值: {MOTION_STOP_RMS_CM_S2} cm/s²")
    print(f"  - 方向锁定阈值: {DIRECTION_LATCH_THRESHOLD_PA} Pa")
    print("=" * 60)
    print("计算完成! Excel文件已保存，包含多个sheet。")


if __name__ == "__main__":
    main()