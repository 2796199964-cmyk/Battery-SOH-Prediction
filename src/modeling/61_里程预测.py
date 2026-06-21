import glob
import os
import pandas as pd
from datetime import timedelta, date
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import shutil
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义参数
y = 0    # 增减SOH(尽量不调）
t = -150  # 手动调整所处周期
k = 1  # 加速衰减系数

# 定义文件夹
source_folder = r'F:\妙算榜-赛题信息\CATL_容量_中值滤波_SOH'
csv_files = glob.glob(os.path.join(source_folder, '0001' '*.csv'))
output_folders = {
    '5': os.path.join(os.path.dirname(source_folder) , 'soh_predicted'),
    '6': os.path.join(os.path.dirname(source_folder) , 'soh_predicted')
}

# 读取模型数据
model_file_path = r'd:\Desktop\CATL\sort_csv_data4\soh预测\soh_predicted\m模型.xlsx'
model_sheets = ['5', '6']

# 对于每个模型执行预测
for sheet in model_sheets:

    # 读取当前模型的数据
    model_df = pd.read_excel(model_file_path, sheet_name=sheet)

    # 确保输出文件夹存在
    output_folder = output_folders[sheet]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 对于每个CSV文件执行数据处理
    for file_path in csv_files:

        # 自动备份当前Python脚本文件
        script_name = os.path.basename(__file__)
        na = r'd:\Desktop\CATL\sort_csv_data4\5d_csv\soh_predicted'
        backup_script_folder = os.path.join(na, 'backup_py')
        if not os.path.exists(backup_script_folder):
            os.makedirs(backup_script_folder)
        backup_script_path = os.path.join(backup_script_folder, f"{os.path.basename(file_path)[:2]}_{script_name}")
        shutil.copy(__file__, backup_script_path)
        #print(f"已备份脚本文件: {backup_script_path}")

        base_name = os.path.basename(file_path)  # 定义base_name

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 转换datatime列为日期时间类型，并仅保留日期
        data['datatime'] = pd.to_datetime(data['datatime']).dt.date

        # 确保datatime列是日期时间格式，并设置为索引
        data['datatime'] = pd.to_datetime(data['datatime'])
        data.set_index('datatime', inplace=True)

        # 重采样为周频率并计算平均值
        weekly_avg = data.resample('W').mean()

        # 检查并删除包含NaN的行
        weekly_avg = weekly_avg.dropna()

        # 准备训练数据
        if not weekly_avg.empty:
            X_train = (weekly_avg.index - weekly_avg.index[0]).days.values.reshape(-1, 1)  # 将日期转换为天数差
            y_train = weekly_avg['cycle'].values

            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 预测未来两年的每周数据
            def daterange(start_date, end_date):
                for n in range(int((end_date - start_date).days)):
                    yield start_date + timedelta(n)

            start_date = data.index.max() + timedelta(days=1)  # 从最后一天开始
            end_date = start_date + timedelta(weeks=104)  # 未来两年
            predicted_dates = list(daterange(start_date, end_date))

            # 创建预测数据
            predicted_weekly_avg = []
            for week in range(104):  # 104周是两年
                days_since_start = (start_date + timedelta(weeks=week) - weekly_avg.index[0]).days
                predicted_value = model.predict([[days_since_start]])[0]
                predicted_weekly_avg.append(predicted_value)

            # 构建预测的DataFrame
            predicted_df = pd.DataFrame({
                'datatime': predicted_dates,
                'cycle': [x for x in predicted_weekly_avg for _ in range(7)]  # 每个周值重复7次代表一周
            })

            # 确保总长度正确
            predicted_df = predicted_df[:len(predicted_dates)]

            # 删除重复数据，这里假设你想要根据datatime去重
            # 如果需要同时考虑cycle，请调整subset参数
            predicted_df = predicted_df.drop_duplicates(subset=['datatime'], keep='first')

            # 查找最接近的SOH值
            def find_closest_soh(cycle):
                closest_cycle = model_df.iloc[((model_df['循环号'] + t) / k - cycle).abs().argsort()[:1]]['SOH_Fit (%)'] - y
                return closest_cycle.iloc[0]

            predicted_df['soh'] = predicted_df['cycle'].apply(find_closest_soh)

            # 将预测数据添加到原始数据中
            combined_df = pd.concat([data, predicted_df.set_index('datatime')], axis=0).sort_index()

            # 保存结果到新的CSV文件
            output_file_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + f'_predicted_{sheet}.csv')
            combined_df.to_csv(output_file_path, index_label='datatime')

            print(f"已创建预测文件: {output_file_path}")

            # 绘制原始数据和预测数据
            plt.figure(figsize=(12, 6))

            # 绘制原始数据
            if 'soh' in data.columns:
                plt.scatter(weekly_avg.index, weekly_avg['soh'], label='Original SOH', color='green', s=5)

            # 绘制预测的SOH数据
            predicted_soh_series = pd.Series(predicted_df['soh'].values,
                                             index=pd.date_range(start=start_date, periods=len(predicted_df), freq='D'))
            plt.plot(predicted_soh_series.index, predicted_soh_series, label=f'Predicted SOH (Model {sheet})', color='orange', linewidth=2)

            # 设置图表标题和标签
            plt.title(f"{os.path.basename(file_path)} SOH Over Time (Model {sheet})")
            plt.xlabel('Date')
            plt.ylabel('SOH (%)')
            plt.legend()

            output_img_predict = os.path.join(output_folder, os.path.splitext(base_name)[0] + f'_predicted_{sheet}.png')
            plt.savefig(output_img_predict)

            # 显示图表
            plt.show()
            plt.close()
        else:
            print(f"文件 {file_path} 中没有有效的数据用于训练模型。")

print("所有文件处理完成。")