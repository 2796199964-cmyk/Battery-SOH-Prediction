import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  # 用于网格搜索
from sklearn.metrics import mean_squared_error  # 用于计算方差
from PyEMD import EMD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor  # 导入 GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # 导入核函数
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

input_folder = r"F:\妙算榜-赛题信息\CATL_容量_中值滤波"
output_folder = r"F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH"
plot_folder =   r"F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH"

def process_csv_files(input_folder, output_folder, plot_folder):
    # 确保输出文件夹和图像文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # 遍历文件夹中的所有CSV文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            plot_file_path = os.path.join(plot_folder, file_name.replace(".csv", ".png"))

            # 读取CSV文件
            df = pd.read_csv(input_file_path)

            # 检查是否存在 'datatime' 列
            if 'datatime' not in df.columns:
                print(f"文件 {file_name} 中缺少 'datatime' 列，跳过处理")
                continue

            # 检查是否存在 'median_filtered_available_capacity' 列
            if 'median_filtered_available_capacity' not in df.columns:
                print(f"文件 {file_name} 中缺少 'median_filtered_available_capacity' 列，跳过处理")
                continue

            # 找到 'median_filtered_available_capacity' 列的最大值
            max_value = df['median_filtered_available_capacity'].max()

            if max_value == 0:
                print(f"文件 {file_name} 中 'median_filtered_available_capacity' 列的最大值为0，跳过计算")
                continue

            # 计算SOH值，并存入新列
            df['SOH'] = df['median_filtered_available_capacity'] / max_value * 100  # 变成百分比

            # 保存修改后的文件到新的文件夹，保留 'datatime' 列
            df.to_csv(output_file_path, index=False)
            print(f"文件 {file_name} 处理完成，已保存至 {output_file_path}")

            # 画图，使用 'datatime' 列作为横坐标
            plt.figure(figsize=(10, 5))
            plt.plot(df['datatime'], df['SOH'], marker='o', linestyle='-')
            plt.xlabel("datatime")  # 使用 'datatime' 列作为横坐标
            plt.ylabel("SOH (%)")  # 纵坐标为SOH
            plt.title(f"{file_name} - SOH变化")
            plt.grid()
            plt.savefig(plot_file_path)
            plt.close()
            print(f"文件 {file_name} 的图像已保存至 {plot_file_path}")

# 设置输入、输出和图像文件夹路径
process_csv_files(input_folder, output_folder, plot_folder)