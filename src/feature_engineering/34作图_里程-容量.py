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

# 定义文件路径
file_path = r'F:\妙算榜-赛题信息\CATL_里程-容量\0001_LW400000003497480.csv'

# 读取保存的CSV文件
data = pd.read_csv(file_path)

# 提取odometer和available_capacity列
odometer = data['odometer']
available_capacity = data['available_capacity']

# 创建Figure对象和子图
fig, ax = plt.subplots(figsize=(10, 6))  # 设置图形大小

# 在子图上绘制折线图
ax.plot(odometer, available_capacity, marker='o', linestyle='-', color='b', label='Available Capacity')
ax.set_title('Available Capacity vs Odometer', fontsize=16)
ax.set_xlabel('Odometer', fontsize=12)
ax.set_ylabel('Available Capacity', fontsize=12)

# 添加网格和图例
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()