import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输入和输出文件夹路径
input_dir = r'F:\妙算榜-赛题信息\CATL_里程预测'
output_dir = r'F:\妙算榜-赛题信息\CATL_里程模型预测'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sheet = '55'

# 获取所有以 '0001' 开头的 CSV 文件
file_list = glob.glob(os.path.join(input_dir, '*.csv'))

# 二分法优化 k 和 best_shift
def binary_search_k_and_shift(x1, x2, k_min, k_max, shift_min, shift_max, tol=1):
    best_k = None
    best_shift = None
    min_error = float('inf')

    while k_max - k_min > tol:
        k_mid = (k_min + k_max) // 2
        x2_scaled = x2 * k_mid

        shift_mid = (shift_min + shift_max) // 2
        x1_shifted = x1 + shift_mid

        # 对齐数据长度
        min_len = min(len(x1_shifted), len(x2_scaled))
        x1_aligned = x1_shifted[:min_len]
        x2_aligned = x2_scaled[:min_len]

        # 计算误差
        error = np.sum((x1_aligned - x2_aligned) ** 2)

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

# 遍历所有符合条件的 CSV 文件
for file_path in file_list:
    print(f"正在处理文件: {file_path}")

    # 读取文件1
    df1 = pd.read_csv(file_path)
    x1 = df1['odometer'].values
    y1 = df1['SOH'].values

    # 读取文件2
    df2 = pd.read_excel(r'd:\Desktop\CATL\s实验室数据\m模型\m模型.xlsx', sheet_name=sheet)
    x2 = df2['循环号'].values
    y2 = df2['SOH_Fit (%)'].values

    # 定义 k 和 shift 的搜索范围
    k_min, k_max = 10, 100000
    shift_min, shift_max = 0, 100000

    # 使用二分法优化 k 和 best_shift
    best_k, best_shift = binary_search_k_and_shift(x1, x2, k_min, k_max, shift_min, shift_max)

    print(f"最佳 k: {best_k}, 最佳平移量: {best_shift}")

    # 使用最佳 k 和 shift 进行数据调整
    x2_scaled = x2 * best_k
    x1_shifted = x1 + best_shift

    # 创建新数据的 DataFrame
    new_data = pd.DataFrame({
        'odometer_shifted': x1_shifted,
        'SOH_shifted': y1
    })

    # 将新数据添加到原始 DataFrame
    df1 = pd.concat([df1, new_data], axis=1)

    # 保存更新后的 CSV 文件
    output_csv_path = os.path.join(output_dir, os.path.basename(file_path))
    df1.to_csv(output_csv_path, index=False)
    print(f"更新后的数据已保存到: {output_csv_path}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x1_shifted, y1, label=f'x1y1 (平移后, shift={best_shift}, k={best_k})', color='blue', s=10)
    plt.plot(x2_scaled, y2, label='x2_scaled y2 (调整后)', color='red', linestyle='--')
    plt.xlabel('Odometer / 循环号')
    plt.ylabel('SOH / SOH_Fit (%)')
    plt.title(f'x1y1 和 x2_scaled y2 对比（平移量: {best_shift}, k={best_k})')
    plt.legend()
    plt.grid(True)

    # 保存图表
    file_name = os.path.basename(file_path).replace('.csv', f'_{sheet}.png')
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")

    # 关闭图表
    plt.close()