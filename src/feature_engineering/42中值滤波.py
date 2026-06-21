import glob
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置TkAgg后端
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# 设置字体和后端
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义源目录、目标目录和图像目录
source_dir = r'F:\妙算榜-赛题信息\CATL_容量_去除异常值'
target_dir = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波'
image_dir = os.path.join(target_dir, '中值滤波前后对比图像')

# 使用 glob 获取所有以 '009' 开头的 CSV 文件
csv_files = glob.glob(os.path.join(source_dir, '0*.csv'))

# 如果目标目录或图像目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 中值滤波函数
def apply_median_filter(data, size=3):
    """
    对数据应用中值滤波。
    :param data: 输入的数据序列（一维数组或列表）
    :param size: 滑动窗口大小（必须为奇数）
    :return: 滤波后的数据序列
    """
    return median_filter(data, size=size)

# 设置中值滤波参数
window_size = 51  # 滑动窗口大小，必须为奇数

# 遍历所有符合条件的 CSV 文件
for filename in csv_files:
    if filename.endswith('.csv'):
        try:
            # 读取CSV文件，确保 datatime 列被正确加载
            data = pd.read_csv(filename)

            # 检查是否存在 datatime 列
            if 'datatime' not in data.columns:
                print(f"文件 {filename} 中缺少 'datatime' 列，跳过处理。")
                continue

            # 提取 available_capacity 列
            available_capacity = data['available_capacity'].values

            # 应用中值滤波
            filtered_available_capacity = apply_median_filter(available_capacity, size=window_size)

            # 将滤波后的结果更新回 DataFrame
            data['median_filtered_available_capacity'] = filtered_available_capacity

            # 构造目标文件路径
            target_file_path = os.path.join(target_dir, os.path.basename(filename))

            # 保存处理后的数据到目标目录，保留 datatime 列
            data.to_csv(target_file_path, index=False)
            print(f"已处理并保存: {target_file_path}")

            # 绘制图像
            plt.figure(figsize=(30, 20))
            plt.plot(data['datatime'], available_capacity, label='原始数据', alpha=0.5)
            plt.plot(data['datatime'], filtered_available_capacity, label='中值滤波', linewidth=2)
            plt.title(f'中值滤波 - {os.path.basename(filename)}')
            plt.xlabel('datatime')
            plt.ylabel('Available Capacity')
            plt.legend()

            # 保存图像
            image_file_path = os.path.join(image_dir, f'{os.path.splitext(os.path.basename(filename))[0]}.png')
            plt.savefig(image_file_path)
            plt.close()
            print(f"已生成并保存图像: {image_file_path}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

print("所有文件的中值滤波及图像生成已完成！")