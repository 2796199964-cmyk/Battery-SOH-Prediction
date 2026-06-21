import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  # 用于网格搜索
from sklearn.metrics import mean_squared_error  # 用于计算方差
from PyEMD import EMD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor  # 导入 GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # 导入核函数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 超参数配置区 ==================
# 数据划分比例
train_ratio = 0.8  # 训练集占总数据的比例

# GPR 模型超参数
gpr_kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-1, 1e2)) \
             + WhiteKernel(noise_level=2, noise_level_bounds=(1e-6, 1e+2))

# 时间步长和未来预测范围
look_back = 50  # 时间步长（回看的数据点数）
future_range = 500000  # 预测未来里程数范围（单位：公里）
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
# 定义输入和输出文件夹路径
input_dir = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH'
output_dir = r'F:\妙算榜-赛题信息\CATL_容量预测_GPR'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有以 '0001' 开头的 CSV 文件
file_list = glob.glob(os.path.join(input_dir, '*.csv'))

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

    # 经验模态分解（EMD）
    emd = EMD()
    imfs = emd.emd(capacity)  # 分解得到多个模态（IMFs）

    # 初始化存储预测结果
    total_future_pred = np.zeros((len(np.arange(0, future_range, future_step)),))  # 总预测结果

    # 计算子图的行列数
    num_imfs = len(imfs)
    rows = (num_imfs + 1) // 2 + ((num_imfs + 1) % 2)  # 行数
    cols = 2  # 列数

    # 创建一个共享的绘图窗口
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))  # 每行两个子图

    # 对每个模态分别进行预测
    for i, imf in enumerate(imfs):
        print(f"处理第 {i + 1} 个模态...")

        # 数据归一化
        scaler_x = MinMaxScaler(feature_range=scaler_x_range)
        scaler_y = MinMaxScaler(feature_range=scaler_y_range)

        odometer_scaled = scaler_x.fit_transform(odometer)
        imf_scaled = scaler_y.fit_transform(imf.reshape(-1, 1))

        # 划分训练集和测试集
        train_size = int(len(odometer_scaled) * train_ratio)
        X_train = odometer_scaled[:train_size]  # 训练集里程数
        y_train = imf_scaled[:train_size]  # 训练集 IMF 值
        X_val = odometer_scaled[train_size:]  # 测试集里程数
        y_val = imf_scaled[train_size:]  # 测试集 IMF 值

        # 构建 GPR 模型
        model = GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10, alpha=0.1)
        model.fit(X_train, y_train.ravel())  # 训练模型

        # 验证模型
        y_val_pred_scaled, y_val_std = model.predict(X_val, return_std=True)
        y_val_pred_scaled = y_val_pred_scaled.reshape(-1, 1)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)  # 反归一化预测结果

        # 预测未来 500,000 公里
        last_odometer_val = odometer[-1]  # 获取最后一个验证集的里程数
        future_odometer = np.arange(
            last_odometer_val + future_step,
            last_odometer_val + future_range + future_step,
            future_step
        ).reshape(-1, 1)

        future_odometer_scaled = scaler_x.transform(future_odometer)  # 归一化未来的里程数
        future_imf_pred_scaled, future_std = model.predict(future_odometer_scaled, return_std=True)
        future_imf_pred_scaled = future_imf_pred_scaled.reshape(-1, 1)
        future_imf_pred = scaler_y.inverse_transform(future_imf_pred_scaled)  # 反归一化预测结果

        future_imf_pred = future_imf_pred.flatten()  # 转换为一维数组

        # 累加预测结果
        total_future_pred += future_imf_pred

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
        ax.set_title(f'预测结果 - IMF {i + 1}')
        ax.set_xlabel('里程数 (odometer)')
        ax.set_ylabel('可用容量 (SOH)')
        ax.legend()
        ax.grid(True)

    # 在最后一个子图中绘制最终叠加结果
    final_ax_idx = num_imfs
    row_idx = final_ax_idx // cols
    col_idx = final_ax_idx % cols
    final_ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

    # 连接原始数据最后一个点和预测未来的第一个点
    last_odometer = odometer[-1]
    last_capacity = capacity[-1]
    first_future_odometer = future_odometer[0]
    first_future_capacity = total_future_pred[0]

    final_ax.scatter(odometer, capacity, color='blue', label='原始数据', alpha=0.5)
    # 绘制连接线
    final_ax.plot(
        [last_odometer, first_future_odometer],
        [last_capacity, first_future_capacity],
        color='red'
    )
    final_ax.plot(future_odometer, total_future_pred, color='red', label='未来预测值（叠加）')
    final_ax.set_title('最终预测结果(GPR) - 叠加')
    final_ax.set_xlabel('里程数 (odometer)')
    final_ax.set_ylabel('可用容量 (SOH)')
    final_ax.legend()
    final_ax.grid(True)

    # 删除多余的子图（如果存在）
    if rows * cols > num_imfs + 1:
        for i in range(num_imfs + 1, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            fig.delaxes(axes[row_idx, col_idx] if rows > 1 else axes[col_idx])

    # 调整子图间距
    plt.tight_layout()

    # 保存图表
    file_name = os.path.basename(file_path).replace('.csv', '.png')
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")
    #plt.show()
    plt.close()

    # 保存预测结果为CSV文件
    future_odometer = np.arange(
        last_odometer_val + future_step,
        last_odometer_val + future_range + future_step,
        future_step
    ).reshape(-1, 1)

    # 创建包含未来里程数和预测容量的DataFrame
    future_predictions = pd.DataFrame({
        'datatime': data['datatime'].iloc[-1],  # 保留原数据的 datatime
        'odometer': future_odometer.flatten(),
        'predicted_SOH': total_future_pred.flatten()
    })

    # 保存为CSV文件
    csv_output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_predictions.csv'))
    future_predictions.to_csv(csv_output_path, index=False)
    print(f"预测结果已保存到: {csv_output_path}")