import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # 导入Keras的Sequential模型
from tensorflow.keras.layers import LSTM, Dense  # 导入LSTM和Dense层
from PyEMD import EMD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输入和输出文件夹路径
input_dir = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波'
output_dir= r'F:\妙算榜-赛题信息\CATL_容量预测_LSTM'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_list = glob.glob(os.path.join(input_dir, '*.csv'))

# ================== 超参数配置区 ==================
train_ratio = 0.6  # 训练集占总数据的比例
# LSTM 模型超参数
lstm_units = 50  # LSTM层的神经元数量
epochs = 50  # 训练轮数
batch_size = 32  # 批量大小

# 时间步长和未来预测范围
look_back = 10  # 时间步长（回看的数据点数）
future_step = 1000  # 预测步长（单位：公里）

# 数据归一化范围
scaler_x_range = (0, 1)  # 输入特征的归一化范围
scaler_y_range = (0, 1)  # 目标变量的归一化范围

# ================== 辅助函数 ==================
def create_dataset(data, look_back):
    """ 将时间序列数据转换为监督学习问题 """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# ================== 主程序区 ==================

# 遍历所有符合条件的 CSV 文件
for file_path in file_list:
    print(f"正在处理文件: {file_path}")

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 提取 odometer 和 median_filtered_available_capacity 列
    odometer = data['odometer'].values.reshape(-1, 1)  # 确保是二维数组
    capacity = data['median_filtered_available_capacity'].values

    # 经验模态分解（EMD）
    emd = EMD()
    imfs = emd.emd(capacity)  # 分解得到多个模态（IMFs）

    # 初始化存储预测结果
    future_range = int((odometer[-1] - odometer[0]) * 0.2)  # 预测未来里程数范围为原始数据的20%
    total_future_pred = np.zeros((len(np.arange(0, future_range, future_step)),))  # 总预测结果

    # 计算子图的行列数
    num_imfs = len(imfs)
    rows = (num_imfs + 1) // 2 + ((num_imfs + 1) % 2)  # 行数
    cols = 2  # 列数

    # 创建一个共享的绘图窗口
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))  # 每行两个子图

    # 对每个模态分别进行预测
    for i, imf in enumerate(imfs):
        print(f"处理第 {i + 1} 个模态...")

        # 数据归一化
        scaler_x = MinMaxScaler(feature_range=scaler_x_range)
        scaler_y = MinMaxScaler(feature_range=scaler_y_range)

        odometer_scaled = scaler_x.fit_transform(odometer)
        imf_scaled = scaler_y.fit_transform(imf.reshape(-1, 1))

        # 创建时间序列数据集
        X, y = create_dataset(imf_scaled, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM 输入形状为 [样本数, 时间步长, 特征数]

        # 划分训练集和测试集
        train_size = int(len(X) * train_ratio)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # 构建 LSTM 模型
        model = Sequential()
        model.add(LSTM(units=lstm_units, input_shape=(look_back, 1)))  # LSTM层
        model.add(Dense(1))  # 输出层
        model.compile(optimizer='adam', loss='mse')  # 编译模型

        # 训练模型
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # 验证模型
        y_val_pred_scaled = model.predict(X_val)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)

        # 计算方差
        variance = np.var(scaler_y.inverse_transform(y_val.reshape(-1, 1)) - y_val_pred)
        print(f"LSTM 方差: {variance}")

        # 预测未来 20% 的里程数
        future_odometer = np.arange(
            odometer[-1],
            odometer[-1] + future_range,
            future_step
        ).reshape(-1, 1)

        # 使用最后一个窗口作为初始输入进行递归预测
        future_input = imf_scaled[-look_back:].reshape(1, look_back, 1)
        future_imf_pred = []

        for step in range(len(future_odometer)):
            # 预测下一个时间步的值
            next_pred_scaled = model.predict(future_input)[0][0]
            next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0][0]
            future_imf_pred.append(next_pred)

            # 更新输入窗口
            next_pred_reshaped = np.array([[next_pred_scaled]])
            future_input = np.concatenate([future_input[:, 1:, :], next_pred_reshaped.reshape(1, 1, 1)], axis=1)

        future_imf_pred = np.array(future_imf_pred).reshape(-1, 1)

        # 累加预测结果
        total_future_pred += future_imf_pred.flatten()

        # 获取当前子图位置
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

        # 在当前子图中绘制 IMF 的结果
        ax.scatter(odometer, imf, color='blue', label=f'原始数据 - IMF {i + 1}', alpha=0.5)
        ax.scatter(
            scaler_x.inverse_transform(odometer_scaled[len(imf_scaled) - len(y_val):]),
            scaler_y.inverse_transform(y_val.reshape(-1, 1)),
            color='green',
            label=f'验证数据 - IMF {i + 1}',
            alpha=0.7
        )
        ax.plot(
            scaler_x.inverse_transform(odometer_scaled[len(imf_scaled) - len(y_val):]),
            y_val_pred,
            color='orange',
            label=f'验证数据预测值 - IMF {i + 1}'
        )
        ax.plot(future_odometer, future_imf_pred, color='red', label=f'未来预测值 - IMF {i + 1}')
        ax.set_title(f'LSTM预测结果 - IMF {i + 1}')
        ax.set_xlabel('里程数 (odometer)')
        ax.set_ylabel('可用容量 (median_filtered_available_capacity)')
        ax.legend()
        ax.grid(True)

    # 在最后一个子图中绘制最终叠加结果
    final_ax_idx = num_imfs
    row_idx = final_ax_idx // cols
    col_idx = final_ax_idx % cols
    final_ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

    final_ax.scatter(odometer, capacity, color='blue', label='原始数据', alpha=0.5)
    final_ax.plot(future_odometer, total_future_pred, color='red', label='未来预测值（叠加）')
    final_ax.set_title('LSTM最终预测结果 - 叠加')
    final_ax.set_xlabel('里程数 (odometer)')
    final_ax.set_ylabel('可用容量 (median_filtered_available_capacity)')
    final_ax.legend()
    final_ax.grid(True)

    # 删除多余的子图（如果存在）
    if rows * cols > num_imfs + 1:
        for i in range(num_imfs + 1, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            fig.delaxes(axes[row_idx, col_idx] if rows > 1 else axes[col_idx])
    plt.tight_layout()
    file_name = os.path.basename(file_path).replace('.csv', '.png')
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")
    #plt.show()