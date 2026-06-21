import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')  # 设置TkAgg后端
import matplotlib.pyplot as plt
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输入和输出目录
input_dir = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH'
output_dir = r'F:\妙算榜-赛题信息\CATL_里程预测'

# 获取所有以 '0001' 开头的 CSV 文件
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

# 检查是否有文件
if not csv_files:
    raise FileNotFoundError(f"No files found in {input_dir} matching the pattern '*.csv'")

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个文件
for file_path in csv_files:
    # 1. 读取数据
    data = pd.read_csv(file_path)

    # 2. 数据预处理
    data['datatime'] = pd.to_datetime(data['datatime'])  # 使用正确的列名 'datatime'
    data['odometer'] = pd.to_numeric(data['odometer'])  # 假设列名是 'odometer'

    # 将日期转换为数值（天数）
    data['days'] = (data['datatime'] - data['datatime'].min()).dt.days

    # 3. 线性拟合
    X = data[['days']]
    y = data['odometer']
    model = LinearRegression()
    model.fit(X, y)

    # 4. 预测未来两年的每周 odometer 值
    last_date = data['datatime'].max()
    future_dates = pd.date_range(start=last_date, periods=104, freq='W')  # 未来两年每周
    future_days = (future_dates - data['datatime'].min()).days.values.reshape(-1, 1)
    future_days_df = pd.DataFrame(future_days, columns=['days'])  # 转换为 DataFrame 并指定列名
    future_odometer = model.predict(future_days_df)

    # 确保预测值的最小值大于或等于原始数据的最大值
    min_future_odometer = future_odometer.min()
    max_historical_odometer = data['odometer'].max()
    if min_future_odometer < max_historical_odometer:
        future_odometer = future_odometer + (max_historical_odometer - min_future_odometer)

    # 创建预测结果的 DataFrame，并保留所有列
    future_data = pd.DataFrame({'datatime': future_dates, 'odometer': future_odometer})

    # 合并历史数据和预测数据，保留所有列
    combined_data = pd.concat([data, future_data], ignore_index=True)

    # 5. 保存结果到 CSV
    output_csv_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.csv', '_predictions.csv')}")
    combined_data.to_csv(output_csv_path, index=False)
    print(f"预测结果已保存到: {output_csv_path}")

    # 6. 绘制图表
    plt.figure(figsize=(50, 30))
    plt.plot(data['datatime'], data['odometer']/1e4, label='历史数据')  # 纵坐标转换为1e4单位
    plt.plot(future_data['datatime'], future_data['odometer']/1e4, label='预测数据', linestyle='--')  # 纵坐标转换

    # 调整x轴格式为月份显示
    plt.gca().xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))

    plt.xlabel('日期')
    plt.ylabel('里程表读数 ( 万km)')  # 更新y轴标签说明

    # 禁用科学计数法，并设置纵坐标格式
    plt.ticklabel_format(axis='y', style='plain')  # 确保没有启用科学计数法
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))  # 设置纵坐标为整数格式

    plt.title(f'未来两年里程表读数预测 - {os.path.basename(file_path)}')
    plt.legend()
    plt.grid(True)

    # 保存图表
    output_plot_path = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.csv', '_prediction_plot.png')}")
    plt.savefig(output_plot_path)
    # print(f"图表已保存到: {output_plot_path}")

    plt.show()  # 如果需要显示图表，可以取消注释此行
    plt.close()