import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 设置TkAgg后端
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输入和输出文件夹路径
source_folder = r'F:\妙算榜-赛题信息\CATL_积分容量'
output_folder = r'F:\妙算榜-赛题信息\CATL_里程-容量'

# 确保输出目录存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有以 '005' 开头的 CSV 文件
file_list = glob.glob(os.path.join(source_folder, '009*.csv'))

# 遍历所有符合条件的 CSV 文件
for input_file in file_list:
    # 读取CSV文件
    data = pd.read_csv(input_file)

    # 筛选条件：charge_state等于1且bat_user_soc_hvs等于80
    filtered_data = data[(data['charge_state'] == 1) & (data['bat_user_soc_hvs'] == 80)]

    # 提取odometer和available_capacity两列
    result_data = filtered_data[['datatime', 'odometer', 'available_capacity']]

    # 获取文件名（不含路径）
    file_name = os.path.basename(input_file)

    # 构建新的文件路径
    output_file = os.path.join(output_folder, file_name)

    # 将结果保存为新的CSV文件
    result_data.to_csv(output_file, index=False)
    print(f"数据已成功提取并保存到 {output_file}")

    # 读取保存的CSV文件
    data = pd.read_csv(output_file)
    # 提取odometer和available_capacity列
    odometer = data['odometer']
    available_capacity = data['available_capacity']

    # 创建Figure对象和子图
    fig, ax = plt.subplots(figsize=(30,20))  # 设置图形大小

    # 在子图上绘制折线图
    ax.plot(odometer, available_capacity, marker='o', linestyle='-', color='b', label='Available Capacity' , linewidth=1 , markersize=2)

    # 添加标题和轴标签
    ax.set_title('Available Capacity vs Odometer', fontsize=16)
    ax.set_xlabel('Odometer', fontsize=12)
    ax.set_ylabel('Available Capacity', fontsize=12)

    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    # 保存图像
    image_dir = r'F:\妙算榜-赛题信息\CATL_里程-容量\图像'
    image_file_path = os.path.join(image_dir, f'{os.path.splitext( input_file)[0]}.png')
    plt.savefig(image_file_path)
    #plt.show()
