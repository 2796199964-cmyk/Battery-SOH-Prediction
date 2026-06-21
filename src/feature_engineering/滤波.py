import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

valid_indices = (y >= 30) & (y <= 70)

# 定义文件夹路径
input_folder = r"F:\妙算榜-赛题信息\CATL_里程-容量"
output_folder = r"F:\妙算榜-赛题信息\CATL_滤波后数据"
output_images_folder = r"F:\妙算榜-赛题信息\CATL_滤波后数据\滤波后图像"
output_gaussian_folder = r"F:\妙算榜-赛题信息\CATL_滤波后数据\高斯滤波后的图像"  # 新增指数滤波图像存放

# 如果输出文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_gaussian_folder, exist_ok=True)


# 三西格玛滤波函数
def three_sigma_filter(y):
    """
    三西格玛滤波函数
    :param y: 输入数据（一维数组或 Pandas Series）
    :return: 过滤后的数据
    """
    y = y.copy().reset_index(drop=True)  # 重置索引，防止 KeyError

    mean = np.mean(y)
    std = np.std(y)

    lower_bound = mean - 0.5 * std
    upper_bound = mean + 0.5 * std

    # 识别异常值（转换为 NumPy 数组避免索引问题）
    is_outlier = (y.values < lower_bound) | (y.values > upper_bound)

    y_filtered = y.copy()
    for i in range(len(y_filtered)):  # 遍历索引
        if is_outlier[i]:
            prev_value = y_filtered[i - 1] if i > 0 else y_filtered[i]
            next_value = y_filtered[i + 1] if i < len(y_filtered) - 1 else y_filtered[i]
            y_filtered[i] = (prev_value + next_value) / 2

    return y_filtered


# 指数平滑滤波（Exponential Moving Average, EMA）
def exponential_moving_average(y, alpha=0.1):
    """
    指数移动平均（EMA）
    :param y: 输入数据（一维数组或 Pandas Series）
    :param alpha: 平滑因子（0~1），值越小越平滑
    :return: 平滑后的数据
    """
    y_smoothed = np.zeros_like(y)
    y_smoothed[0] = y[0]  # 初始化第一个值

    for i in range(1, len(y)):
        y_smoothed[i] = alpha * y[i] + (1 - alpha) * y_smoothed[i - 1]

    return y_smoothed

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith("005*.csv"):
        file_path = os.path.join(input_folder, file_name)

        # 读取 CSV 文件
        data = pd.read_csv(file_path)

        # 确保数据至少有两列
        if data.shape[1] < 2:
            print(f"文件 {file_name} 格式错误，跳过处理。")
            continue

        # 第一列是x轴（里程数等），第二列是y轴（容量等）
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]

        # **步骤 1: 先筛选，删除 y < 30 或 y > 100 的行**
        valid_indices = (y >= 30) & (y <= 100)
        x = x[valid_indices].reset_index(drop=True)
        y = y[valid_indices].reset_index(drop=True)

        if len(y) == 0:
            print(f"文件 {file_name} 所有数据均被筛除，跳过处理。")
            continue

        # **步骤 2: 进行三西格玛滤波**
        y_filtered = three_sigma_filter(y)

        # **步骤 3: 再进行指数滤波**
        y_smoothed = exponential_moving_average(y_filtered)

        # **步骤 4: 保存滤波后的数据**
        filtered_data = pd.DataFrame({data.columns[0]: x, data.columns[1]: y_smoothed})
        output_file_path = os.path.join(output_folder, file_name)
        filtered_data.to_csv(output_file_path, index=False)

        # **步骤 5: 画图**
        plt.figure(figsize=(100, 60))
        plt.plot(x, y, marker="o", linestyle="-", color="b", label="原始数据")
        plt.plot(x, y_filtered, marker="o", linestyle="-", color="r", label="三西格玛滤波")
        plt.plot(x, y_smoothed, marker="o", linestyle="-", color="g", label="指数平滑滤波")
        plt.xlabel("odometer")
        plt.ylabel("容量")
        plt.title(f"{file_name} 数据画图（三西格玛 + 指数滤波）")
        plt.legend()
        plt.grid()

        # **步骤 6: 保存图像**
        image_file_path = os.path.join(output_images_folder, file_name.replace(".csv", ".png"))
        plt.savefig(image_file_path)
        plt.close()  # 关闭图像，避免内存泄漏

        # **步骤 7: 保存指数滤波后的图像**
        plt.figure(figsize=(100, 60))
        plt.plot(x, y_smoothed, marker="o", linestyle="-", color="g", label="指数平滑滤波" )
        plt.xlabel("odometer")
        plt.ylabel("容量")
        plt.title(f"{file_name} 数据画图（指数滤波）")
        plt.legend()
        plt.grid()

        image_gaussian_path = os.path.join(output_gaussian_folder, file_name.replace(".csv", ".png"))
        plt.savefig(image_gaussian_path)
        plt.close()  # 关闭图像，避免内存泄漏

print(f"所有 CSV 文件已处理完毕，滤波后的数据已保存到文件夹: {output_folder}")
print(f"所有图像已保存到文件夹: {output_images_folder}")
print(f"指数滤波后的图像已保存到文件夹: {output_gaussian_folder}")
