import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输入和输出文件夹路径
input_dir = r'F:\妙算榜-赛题信息\CATL_容量预测_XGboost+GPR'
output_dir = r'F:\妙算榜-赛题信息\CATL_容量预测_XGboost+GPR_结果'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有 CSV 文件
file_list = glob.glob(os.path.join(input_dir, '*.csv'))

# 定义评估指标计算函数
def calculate_metrics(y_true, y_pred):
    # 确保 y_true 和 y_pred 都不包含 NaN 值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0 or len(y_pred) == 0:
        print("警告：y_true 或 y_pred 为空，无法计算评估指标。")
        return np.nan, np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# 二分法优化 k 和 best_shift
def binary_search_k_and_shift(x1, y1_pred, x2, y2, k_min, k_max, shift_min, shift_max, tol=1):
    best_k = None
    best_shift = None
    min_error = float('inf')

    # 检查数据是否存在缺失值
    if np.isnan(x1).any() or np.isnan(y1_pred).any() or np.isnan(x2).any() or np.isnan(y2).any():
        print("警告：输入数据存在缺失值，无法进行优化。")
        return best_k, best_shift

    while k_max - k_min > tol:
        k_mid = (k_min + k_max) // 2
        x2_scaled = x2 * k_mid

        shift_mid = (shift_min + shift_max) // 2
        x1_shifted = x1 + shift_mid

        # 对齐数据长度
        min_len = min(len(x1_shifted), len(x2_scaled))
        x1_aligned = x1_shifted[:min_len]
        x2_aligned = x2_scaled[:min_len]
        y1_pred_aligned = y1_pred[:min_len]
        y2_aligned = y2[:min_len]

        # 计算误差（使用 y1_pred 和 y2 的误差作为优化目标）
        error = np.sum((y1_pred_aligned - y2_aligned) ** 2)

        if error < min_error:
            min_error = error
            best_k = k_mid
            best_shift = shift_mid
            k_max = k_mid
            shift_max = shift_mid
        else:
            k_min = k_mid
            shift_min = shift_mid

    return best_k, best_shift

# 遍历所有 CSV 文件
for file_path in file_list:
    print(f"正在处理文件: {file_path}")

    # 读取文件1
    df1 = pd.read_csv(file_path)

    # 检查是否存在 SOH 和 predicted_SOH 列
    if 'SOH' not in df1.columns or 'predicted_SOH' not in df1.columns:
        print(f"文件 {file_path} 中缺少 'SOH' 或 'predicted_SOH' 列，跳过处理。")
        continue

    # 用 predicted_SOH 补充 SOH 的空值
    df1['SOH'].fillna(df1['predicted_SOH'], inplace=True)

    # 提取 x1 和 y1_pred
    x1 = df1['odometer'].values
    y1_pred = df1['predicted_SOH'].values

    # 读取文件2
    df2 = pd.read_excel(r'd:\Desktop\CATL\s实验室数据\m模型\m模型.xlsx', sheet_name='55')
    x2 = df2['循环号'].values
    y2 = df2['SOH_Fit (%)'].values

    # 定义 k 和 shift 的搜索范围
    k_min, k_max = 1, 1000000  # 扩大 k 的范围
    shift_min, shift_max = -100000, 100000  # 扩大 shift 的范围

    # 使用二分法优化 k 和 best_shift
    best_k, best_shift = binary_search_k_and_shift(x1, y1_pred, x2, y2, k_min, k_max, shift_min, shift_max)

    print(f"最佳 k: {best_k}, 最佳平移量: {best_shift}")

    # 使用最佳 k 和 shift 进行数据调整
    x2_scaled = x2 * best_k
    x1_shifted = x1 + best_shift

    # 计算 predicted_SOH 的评估指标
    y_true = df1['SOH'].values  # 真实值
    y_pred = df1['predicted_SOH'].values  # 预测值
    rmse, mae, r2 = calculate_metrics(y_true, y_pred)

    # 如果评估指标为 NaN，跳过可视化
    if np.isnan(rmse) or np.isnan(mae) or np.isnan(r2):
        print(f"文件 {file_path} 的评估指标为 NaN，跳过可视化。")
        continue

    # 可视化
    plt.figure(figsize=(12, 6))

    # 绘制 SOH 和 predicted_SOH
    plt.scatter(x1_shifted, df1['SOH'], label='SOH (真实值)', color='blue', s=10)
    plt.scatter(x1_shifted, df1['predicted_SOH'], label='predicted_SOH (预测值)', color='green', s=10)

    # 绘制 x2_scaled y2
    plt.plot(x2_scaled, y2, label='x2_scaled y2 (调整后)', color='red', linestyle='--')

    # 添加标题和标签
    plt.xlabel('Odometer / 循环号')
    plt.ylabel('SOH / SOH_Fit (%)')
    plt.title(f'SOH 和 predicted_SOH 对比\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}')
    plt.legend()
    plt.grid(True)

    # 保存图表
    file_name = os.path.basename(file_path).replace('.csv', '_对比.png')
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")

    # 关闭图表
    plt.close()

    # 保存更新后的 CSV 文件
    output_csv_path = os.path.join(output_dir, os.path.basename(file_path))
    df1.to_csv(output_csv_path, index=False)
    print(f"更新后的数据已保存到: {output_csv_path}")