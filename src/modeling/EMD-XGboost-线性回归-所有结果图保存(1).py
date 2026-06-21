import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from PyEMD import EMD  # 需要安装 PyEMD 库以进行经验模态分解


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 超参数配置区 ==================
# 文件夹路径
data_folder = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波'

# 输出图片保存路径
output_folder = r'F:\妙算榜-赛题信息\CATL_容量预测_XGboost线性'

# 数据划分比例
train_ratio = 0.8  # 训练集占总数据的比例

# XGBoost 模型超参数
xgb_params = {
    'n_estimators': 100,  # 树的数量
    'learning_rate': 0.1,  # 学习率
    'max_depth': 6,  # 树的最大深度
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse'  # 评估指标
}

# 多项式回归超参数
poly_degree = 1  # 多项式的阶数

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
# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有 CSV 文件
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# 遍历每个 CSV 文件
for file_name in csv_files:
    print(f"处理文件: {file_name}")
    file_path = os.path.join(data_folder, file_name)

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 提取 odometer 和 median_filtered_available_capacity 列
    odometer = data['odometer'].values.reshape(-1, 1)  # 确保是二维数组
    capacity = data['median_filtered_available_capacity'].values

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
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))  # 每行两个子图

    # 对每个模态分别进行预测
    for i, imf in enumerate(imfs):
        print(f"处理第 {i + 1} 个模态...")

        if i == len(imfs) - 1:  # 最后一个 IMF 使用多项式回归
            print("使用多项式回归处理最后一个 IMF...")

            # 数据归一化
            scaler_x = MinMaxScaler(feature_range=scaler_x_range)
            scaler_y = MinMaxScaler(feature_range=scaler_y_range)

            odometer_scaled = scaler_x.fit_transform(odometer)  # 归一化里程数
            imf_scaled = scaler_y.fit_transform(imf.reshape(-1, 1))  # 归一化 IMF 值

            # 划分训练集和测试集
            train_size = int(len(odometer_scaled) * train_ratio)
            X_train = odometer_scaled[:train_size]  # 训练集里程数
            y_train = imf_scaled[:train_size]  # 训练集 IMF 值
            X_val = odometer_scaled[train_size:]  # 测试集里程数
            y_val = imf_scaled[train_size:]  # 测试集 IMF 值

            # 构建多项式回归模型
            poly = PolynomialFeatures(degree=poly_degree)
            X_poly_train = poly.fit_transform(X_train)  # 生成多项式特征
            model = LinearRegression()
            model.fit(X_poly_train, y_train.ravel())  # 训练模型

            # 验证模型
            X_poly_val = poly.transform(X_val)
            y_val_pred_scaled = model.predict(X_poly_val).reshape(-1, 1)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)  # 反归一化预测结果

            # 预测未来 500,000 公里
            # 修改：从验证集最后一个数据点的公里数开始递增
            future_odometer_start = odometer[-1] + future_step
            future_odometer = np.arange(
                future_odometer_start,
                future_odometer_start + future_range,
                future_step
            ).reshape(-1, 1)

            future_odometer_scaled = scaler_x.transform(future_odometer)  # 归一化未来的里程数
            X_poly_future = poly.transform(future_odometer_scaled)  # 生成多项式特征
            future_imf_pred_scaled = model.predict(X_poly_future).reshape(-1, 1)
            future_imf_pred = scaler_y.inverse_transform(future_imf_pred_scaled)  # 反归一化预测结果

            future_imf_pred = future_imf_pred.flatten()  # 转换为一维数组

        else:  # 其他 IMF 使用 XGBoost
            print("使用 XGBoost 处理其他 IMF...")
            # 数据归一化
            scaler_x = MinMaxScaler(feature_range=scaler_x_range)
            scaler_y = MinMaxScaler(feature_range=scaler_y_range)

            odometer_scaled = scaler_x.fit_transform(odometer)
            imf_scaled = scaler_y.fit_transform(imf.reshape(-1, 1))

            # 创建时间序列数据集
            X, y = create_dataset(imf_scaled, look_back)
            X = X.reshape(X.shape[0], -1)  # XGBoost 输入形状为 [样本数, 特征数]

            # 划分训练集和测试集
            train_size = int(len(X) * train_ratio)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # 构建 XGBoost 模型
            model = XGBRegressor(
                n_estimators=xgb_params['n_estimators'],
                learning_rate=xgb_params['learning_rate'],
                max_depth=xgb_params['max_depth'],
                objective=xgb_params['objective'],
                eval_metric=xgb_params['eval_metric']
            )
            model.fit(X_train, y_train.ravel())

            # 验证模型
            y_val_pred_scaled = model.predict(X_val).reshape(-1, 1)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)

            # 预测未来 500,000 公里
            # 修改：从验证集最后一个数据点的公里数开始递增
            future_odometer_start = odometer[-1] + future_step
            future_odometer = np.arange(
                future_odometer_start,
                future_odometer_start + future_range,
                future_step
            ).reshape(-1, 1)

            # 使用最后一个窗口作为初始输入进行递归预测
            future_input = imf_scaled[-look_back:].reshape(1, -1)
            future_imf_pred = []

            for step in range(len(future_odometer)):
                # 预测下一个时间步的值
                next_pred_scaled = model.predict(future_input)[0]
                next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0][0]
                future_imf_pred.append(next_pred)

                # 更新输入窗口
                next_pred_reshaped = np.array([[next_pred_scaled]])
                future_input = np.concatenate([future_input[:, 1:], next_pred_reshaped], axis=1)

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
        ax.set_title(f'预测结果 - IMF {i + 1}')
        ax.set_xlabel('里程数 (odometer)')
        ax.set_ylabel('可用容量 (median_filtered_available_capacity)')
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

    # 绘制原始数据
    final_ax.scatter(odometer, capacity, color='blue', label='原始数据', alpha=0.5)

    # 绘制连接线
    final_ax.plot(
        [last_odometer, first_future_odometer],
        [last_capacity, first_future_capacity],
        color='red'
    )

    # 绘制未来预测值
    final_ax.plot(future_odometer, total_future_pred, color='red', label='未来预测值（叠加）')

    final_ax.set_title('最终预测结果 - 叠加')
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

    # 调整子图间距
    plt.tight_layout()

    # 保存图表为 PDF 文件
    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
    plt.savefig(output_file, format='png', dpi=300)
    print(f"保存图表到: {output_file}")

    # 清空当前图表以便下次绘图
    plt.close(fig)

print("所有文件处理完成！")