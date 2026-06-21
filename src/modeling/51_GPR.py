import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import glob

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 超参数配置区 ==================
# 数据划分比例
train_ratio = 0.8  # 训练集占总数据的比例

# GPR 模型超参数
gpr_kernel = ConstantKernel() * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-6, 1e+1))

# 时间步长和未来预测范围
look_back = 50  # 时间步长（回看的数据点数）
future_range = 500000  # 预测未来里程数范围（单位：公里）
future_step = 1000  # 预测步长（单位：公里）

# 数据归一化范围
scaler_x_range = (0, 1)  # 输入特征的归一化范围
scaler_y_range = (0, 1)  # 目标变量的归一化范围

# ================== 辅助函数 ==================
def create_time_series_features(data, look_back):
    """ 创建时间序列特征 """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)].flatten())  # 展平时间步长
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def recursive_predict(model, initial_input, future_steps):
    """ 递归预测未来数据 """
    predictions = []
    current_input = initial_input.copy()

    for _ in range(future_steps):
        # 预测下一个点
        next_pred_scaled = model.predict(current_input, return_std=False)
        predictions.append(next_pred_scaled[0])

        # 更新输入窗口
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_pred_scaled

    return np.array(predictions)

# ================== 主程序区 ==================
# 定义输入和输出文件夹路径
input_dir = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH'
output_dir = r'F:\妙算榜-赛题信息\CATL_容量预测_单GPR无分解'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有以 '0001' 开头的 CSV 文件
file_list = glob.glob(os.path.join(input_dir, '009*.csv'))

# 遍历所有符合条件的 CSV 文件
for file_path in file_list:
    print(f"正在处理文件: {file_path}")

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 检查是否存在 'datatime' 列
    if 'datatime' not in data.columns:
        print(f"文件 {file_path} 中缺少 'datatime' 列，跳过处理。")
        continue

    # 提取 odometer 和 SOH 列
    odometer = data['odometer'].values.reshape(-1, 1)  # 确保是二维数组
    capacity = data['SOH'].values

    # 数据归一化
    scaler_x = MinMaxScaler(feature_range=scaler_x_range)
    scaler_y = MinMaxScaler(feature_range=scaler_y_range)

    odometer_scaled = scaler_x.fit_transform(odometer)
    capacity_scaled = scaler_y.fit_transform(capacity.reshape(-1, 1))

    # 创建时间序列特征
    X, y = create_time_series_features(capacity_scaled, look_back)

    # 按时间顺序划分训练集和测试集
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 构建 GPR 模型
        model = GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10, alpha=0.1)
        model.fit(X_train, y_train.ravel())

        # 验证模型
        y_val_pred_scaled, y_val_std = model.predict(X_val, return_std=True)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1))

        # 计算均方误差
        mse = mean_squared_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_val_pred)
        print(f"验证集均方误差: {mse}")

    # 预测未来数据
    future_odometer = np.arange(
        odometer[-1] + future_step,
        odometer[-1] + future_range + future_step,
        future_step
    ).reshape(-1, 1)

    # 创建初始输入窗口
    initial_input = capacity_scaled[-look_back:].reshape(1, -1)

    # 递归预测未来数据
    future_steps = len(future_odometer)
    future_capacity_pred_scaled = recursive_predict(model, initial_input, future_steps)
    future_capacity_pred = scaler_y.inverse_transform(future_capacity_pred_scaled.reshape(-1, 1))

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.scatter(odometer, capacity, color='blue', label='原始数据', alpha=0.5)
    plt.plot(future_odometer, future_capacity_pred, color='red', label='未来预测值')
    plt.title('GPR 预测结果')
    plt.xlabel('里程数 (odometer)')
    plt.ylabel('可用容量 (SOH)')
    plt.legend()
    plt.grid(True)

    # 保存图表
    file_name = os.path.basename(file_path).replace('.csv', '.png')
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")
    plt.close()

    # 保存预测结果为CSV文件
    future_predictions = pd.DataFrame({
        'datatime': data['datatime'].iloc[-1],  # 保留原数据的 datatime
        'odometer': future_odometer.flatten(),
        'predicted_SOH': future_capacity_pred.flatten()
    })

    # 保存为CSV文件
    csv_output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_predictions.csv'))
    future_predictions.to_csv(csv_output_path, index=False)
    print(f"预测结果已保存到: {csv_output_path}")