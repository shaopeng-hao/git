"""
振动数据形象对比分析
====================
功能：
  1. 从 Athena 查询三组振动数据
  2. 对比分析：gbvibforwardrms vs addvibforwardrms（向前振动）
              gbviblateralrms vs addviblateralrms（横向振动）
              gbvibverticalrms vs addvibverticalrms（垂直振动）
  3. 生成形象对比图表和统计信息

依赖：numpy, pandas, plotly, kaleido, pyathena, boto3
安装：pip install numpy pandas plotly kaleido pyathena boto3

运行：python vibration_comparison.py
"""

import os
import warnings
import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyathena

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ============================================================
# 配置（改成你要查询的设备和日期）
# ============================================================
EQUIPMENT_NUMBERS = ['45024583']
START_DATE = '2026-09-06'
END_DATE = '2026-09-06'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要对比的振动数据字段
VIBRATION_PAIRS = [
    ("gbvibforwardrms", "addvibforwardrms", "向前振动", "blue", "orange"),
    ("gbviblateralrms", "addviblateralrms", "横向振动", "green", "red"),
    ("gbvibverticalrms", "addvibverticalrms", "垂直振动", "purple", "brown"),
]

# ============================================================
# Athena 连接
# ============================================================
class Athena_Connector:
    def __init__(self):
        self.conn = pyathena.connect(
            s3_staging_dir="s3://athena-query-prod-zhen/",
            session=boto3.Session(profile_name='AWSPowerUserAccess-579289528406')
        )

    def query(self, sql):
        return pd.read_sql(sql, self.conn)

    def close(self):
        self.conn.close()

# ============================================================
# 数据清洗
# ============================================================
def clean_vibration_pair(data1, data2):
    """清洗一对振动数据，只保留两者都有有效值的数据点"""
    valid_mask = np.isfinite(data1) & np.isfinite(data2)
    return data1[valid_mask], data2[valid_mask], valid_mask

# ============================================================
# 对比分析统计函数
# ============================================================
def compare_vibration_stats(data_gb, data_add, pair_name):
    """计算一对振动数据的对比统计"""
    if len(data_gb) == 0 or len(data_add) == 0:
        return None
    
    # 差值分析
    differences = data_add - data_gb
    abs_differences = np.abs(differences)
    
    # 比例分析
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = data_add / data_gb
        ratios = ratios[np.isfinite(ratios)]
    
    stats = {
        "pair_name": pair_name,
        "count": len(data_gb),
        # GB 振动统计
        "gb_mean": np.mean(data_gb),
        "gb_std": np.std(data_gb),
        "gb_min": np.min(data_gb),
        "gb_max": np.max(data_gb),
        # ADD 振动统计
        "add_mean": np.mean(data_add),
        "add_std": np.std(data_add),
        "add_min": np.min(data_add),
        "add_max": np.max(data_add),
        # 差值统计
        "diff_mean": np.mean(differences),
        "diff_std": np.std(differences),
        "abs_diff_mean": np.mean(abs_differences),
        # 比例统计
        "ratio_mean": np.mean(ratios) if len(ratios) > 0 else None,
        "ratio_std": np.std(ratios) if len(ratios) > 0 else None,
        # 关系分析
        "add_greater_count": int(np.sum(data_add > data_gb)),
        "add_greater_percentage": float(np.sum(data_add > data_gb) / len(data_gb) * 100),
        "gb_greater_count": int(np.sum(data_gb > data_add)),
        "gb_greater_percentage": float(np.sum(data_gb > data_add) / len(data_gb) * 100),
        "equal_count": int(np.sum(data_gb == data_add)),
    }
    
    return stats

# ============================================================
# 生成对比图表
# ============================================================
def create_comparison_charts(df, stats_list):
    """生成振动数据对比图表"""
    
    # 创建子图：3行 x 1列，只显示三组GB与ADD振动值对比
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("向前振动对比", "横向振动对比", "垂直振动对比"),
        vertical_spacing=0.12,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}]
        ]
    )
    
    for i, (gb_col, add_col, pair_name, color_gb, color_add) in enumerate(VIBRATION_PAIRS):
        row = i + 1
        
        # 获取数据
        gb_data = df[gb_col].values.astype(float)
        add_data = df[add_col].values.astype(float)
        
        # 清理数据
        gb_clean, add_clean, valid_mask = clean_vibration_pair(gb_data, add_data)
        time_clean = df["event_time_shanghai"].iloc[np.where(valid_mask)[0]]
        
        # 1. 时间序列对比（左列）
        # GB振动数据
        fig.add_trace(
            go.Scatter(
                x=time_clean,
                y=gb_clean,
                mode="lines+markers",
                name=f"{pair_name}-GB",
                marker=dict(size=4, color=color_gb),
                line=dict(width=1.5, color=color_gb),
                hovertemplate=f"时间=%{{x}}<br>GB={pair_name}=%{{y:.4f}}<extra></extra>",
            ),
            row=row, col=1
        )
        
        # ADD振动数据（右侧y轴）
        fig.add_trace(
            go.Scatter(
                x=time_clean,
                y=add_clean,
                mode="lines+markers",
                name=f"{pair_name}-ADD",
                marker=dict(size=4, color=color_add),
                line=dict(width=1.5, color=color_add),
                hovertemplate=f"时间=%{{x}}<br>ADD={pair_name}=%{{y:.4f}}<extra></extra>",
                yaxis=f"y{row*2}" if row > 1 else "y2"
            ),
            row=row, col=1,
            secondary_y=True
        )
        
    # 每一行只保留一组 GB 与 ADD 的时间序列对比图
    
    # 更新布局
    fig.update_layout(
        title=f"设备 {EQUIPMENT_NUMBERS} 振动数据对比分析 ({START_DATE} ~ {END_DATE})",
        height=1000,
        width=1400,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
    )
    
    # 三个子图分别显示一组GB与ADD振动值
    for i in range(1, 4):
        fig.update_yaxes(title_text="GB振动值", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="ADD振动值", row=i, col=1, secondary_y=True)
        fig.update_xaxes(title_text="时间", tickangle=45, row=i, col=1)
    
    return fig

# ============================================================
# 生成统计摘要表格
# ============================================================
def create_statistics_summary(stats_list):
    """生成统计摘要"""
    print("\n" + "="*80)
    print("振动数据对比统计摘要")
    print("="*80)
    
    for stats in stats_list:
        if stats is None:
            continue
            
        print(f"\n{stats['pair_name']}对比分析:")
        print(f"  数据点数: {stats['count']}")
        print(f"  GB振动 - 均值: {stats['gb_mean']:.4f}, 标准差: {stats['gb_std']:.4f}, 范围: [{stats['gb_min']:.4f}, {stats['gb_max']:.4f}]")
        print(f"  ADD振动 - 均值: {stats['add_mean']:.4f}, 标准差: {stats['add_std']:.4f}, 范围: [{stats['add_min']:.4f}, {stats['add_max']:.4f}]")
        print(f"  差值统计 - 均值: {stats['diff_mean']:.4f}, 标准差: {stats['diff_std']:.4f}, 绝对差值均值: {stats['abs_diff_mean']:.4f}")
        
        if stats['ratio_mean'] is not None:
            print(f"  比例统计 - ADD/GB均值: {stats['ratio_mean']:.4f}, 标准差: {stats['ratio_std']:.4f}")
        
        print(f"  关系分析:")
        print(f"    ADD > GB: {stats['add_greater_count']}点 ({stats['add_greater_percentage']:.1f}%)")
        print(f"    GB > ADD: {stats['gb_greater_count']}点 ({stats['gb_greater_percentage']:.1f}%)")
        print(f"    GB = ADD: {stats['equal_count']}点")

# ============================================================
# 原有代码部分（隐藏不运行）
# ============================================================
"""
以下是原有zxc.py代码，已被注释掉，不会被运行。
如果需要使用原有功能，请取消注释并注释掉下面的主程序部分。
"""

"""
def kde_threshold(data: np.ndarray, bandwidth: str = "scott") -> float:
    # KDE：找两峰之间的密度谷底作为阈值
    data = data[np.isfinite(data)]
    if len(data) < 10:
        raise ValueError("数据太少，无法做 KDE")

    x = data.reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(x)

    grid = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    density = np.exp(kde.score_samples(grid))
    grid_x = grid[:, 0]

    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] >= density[i + 1]:
            peaks.append((grid_x[i], density[i]))

    if len(peaks) < 2:
        return _otsu_fallback(data)

    peaks.sort(key=lambda p: p[1], reverse=True)
    lo, hi = min(peaks[0][0], peaks[1][0]), max(peaks[0][0], peaks[1][0])

    valleys = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            if lo <= grid_x[i] <= hi:
                valleys.append((grid_x[i], density[i]))

    if not valleys:
        return _otsu_fallback(data)

    valleys.sort(key=lambda v: v[1])
    return float(valleys[0][0])


def _otsu_fallback(data: np.ndarray, bins: int = 256) -> float:
    # KDE 失败时的��退：大津算法
    data = data[np.isfinite(data)]
    hist, edges = np.histogram(data, bins=bins)
    total = len(data)
    best_var, best_t = 0, edges[0]
    for i in range(bins - 1):
        w0 = np.sum(hist[:i+1])
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        bc0 = (edges[:i+1] + edges[1:i+2]) / 2
        m0 = np.sum(hist[:i+1] * bc0) / w0
        bc1 = (edges[i+1:bins] + edges[i+2:bins+1]) / 2
        m1 = np.sum(hist[i+1:bins] * bc1) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2 / (total ** 2)
        if var_between > best_var:
            best_var = var_between
            best_t = edges[i+1]
    return float(best_t)
"""

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"设备: {EQUIPMENT_NUMBERS}")
    print(f"日期: {START_DATE} ~ {END_DATE}")
    
    # 1. 查询数据（包含所有振动字段）
    print("\n正在从 Athena 查询数据……")
    eq_list = ", ".join(f"'{e}'" for e in EQUIPMENT_NUMBERS)
    
    # 构建查询字段列表
    vib_fields = []
    for gb_col, add_col, _, _, _ in VIBRATION_PAIRS:
        vib_fields.append(gb_col)
        vib_fields.append(add_col)
    
    sql = f"""
        SELECT equipmentnumber, {', '.join(vib_fields)}, 
               motorcurrent1avg, stepbandspeedleftavg,
               modeset, operationstatus,
               (FROM_ISO8601_TIMESTAMP(timestamp) AT TIME ZONE 'Asia/Shanghai') AS event_time_shanghai
        FROM data_cleansed."anyescalator"
        WHERE equipmentnumber IN ({eq_list})
          AND eventdate BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY timestamp
    """
    
    conn = Athena_Connector()
    try:
        combined = conn.query(sql)
    finally:
        conn.close()
    
    print(f"查询到 {len(combined)} 条数据")
    
    # 去重
    dedup_cols = ['equipmentnumber', 'event_time_shanghai']
    combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    
    # 2. 生成三组振动值对比图
    print("\n正在生成三组振动值对比图……")
    fig = create_comparison_charts(combined, None)
    
    # 5. 保存结果
    eq_tag = "_".join(EQUIPMENT_NUMBERS)
    output_dir = os.path.join(OUTPUT_DIR, f"{eq_tag}_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f"{eq_tag}_comparison.csv")
    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 已保存: {csv_path}")
    
    # 保存HTML交互图表
    html_path = os.path.join(output_dir, f"{eq_tag}_comparison.html")
    fig.write_html(html_path)
    print(f"HTML交互图表已保存: {html_path}")
    
    # 尝试保存PNG（需要kaleido）
    try:
        png_path = os.path.join(output_dir, f"{eq_tag}_comparison.png")
        fig.write_image(png_path, width=1600, height=1200, scale=2)
        print(f"静态图片已保存: {png_path}")
    except Exception as e:
        print(f"[提示] 静态图片导出失败，需要安装 kaleido: pip install kaleido")
        print(f"  错误: {e}")
    
    # 显示图表
    print("\n图表已生成，正在显示……")
    fig.show()
    
    print("\n对比分析完成！")