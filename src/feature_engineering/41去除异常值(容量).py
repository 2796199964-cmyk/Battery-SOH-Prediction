import glob
import pandas as pd
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置TkAgg后端
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用 glob 获取所有以 '0001' 开头的 CSV 文件
csv_files = glob.glob(os.path.join(source_dir, '009*.csv'))
source_dir = r'F:\妙算榜-赛题信息\CATL_里程-容量'
target_dir = r'F:\妙算榜-赛题信息\CATL_容量_去除异常值'
image_dir = os.path.join(target_dir, '容量_去除异常值图像')

max_cap = 70
min_cap = 45

# 如果目标目录或图像目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


for filename in csv_files:  # 直接遍历匹配到的文件

    # 备份当前运行的Python文件
    current_py_file = os.path.basename(__file__)
    backup_py_file_name = f"{os.path.splitext(os.path.basename(filename))[0]}_区间{min_cap}-{max_cap}_{current_py_file}"
    backup_py_file_path = os.path.join(target_dir, backup_py_file_name)
    shutil.copyfile(__file__, backup_py_file_path)
    #print(f"当前py文件已备份到: {backup_py_file_path}")

    if filename.endswith('.csv'):
        file_path = filename  # 现在filename已经是完整路径

        try:
            # 读取数据并过滤
            data = pd.read_csv(file_path)
            filtered_data = data[
                (data['available_capacity'].notna()) &
                (data['available_capacity'] <= max_cap) &
                (data['available_capacity'] >= min_cap)
                ]

            target_file_path = os.path.join(target_dir, os.path.basename(filename))
            filtered_data.to_csv(target_file_path, index=False)
            print(f"已处理并保存: {target_file_path}")

            # 绘制图像
            plt.figure(figsize=(10, 6))  # 调整图像大小以便于查看
            sns.histplot(filtered_data['available_capacity'], kde=True)
            plt.title(f'Available Capacity Distribution - {os.path.basename(filename)}')
            plt.xlabel('Available Capacity')
            plt.ylabel('Frequency')

            # 保存图像
            image_file_path = os.path.join(image_dir, f'{os.path.splitext(os.path.basename(filename))[0]}.png')
            plt.savefig(image_file_path)
            plt.close()
            #print(f"已生成并保存图像: {image_file_path}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

print("所有文件处理及图像生成完成！")