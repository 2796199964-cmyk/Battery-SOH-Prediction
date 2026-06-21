import pandas as pd
import os
import glob
import shutil

# 定义目录路径
source_folder = r"F:\妙算榜-赛题信息\CATL_充电指定SOC区间"
output_folder = r'F:\妙算榜-赛题信息\CATL_积分容量'

# 确保输出目录存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# 获取所有以 '005' 开头的 CSV 文件
file_list = glob.glob(os.path.join(source_folder, '*.csv'))

# 遍历所有符合条件的 CSV 文件
for file_path in file_list:
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 备份当前运行的Python文件
    current_py_file = os.path.basename(__file__)
    backup_py_file_name = f"{os.path.splitext(file_path)[0]}_{current_py_file}"
    backup_py_file_path = os.path.join(output_folder, backup_py_file_name)
    shutil.copyfile(__file__, backup_py_file_path)
    print(f"当前py文件已备份到: {backup_py_file_path}")

    # 转换时间戳列为datetime类型
    data['datatime'] = pd.to_datetime(data['datatime'])

    # 按时间戳排序
    data = data.sort_values('datatime').reset_index(drop=True)

    # 只保留充电状态为1、2、3、4的行
    charging_data = data[data['charge_state'].isin([1, 2, 3, 4])]

    # 初始化可用容量列
    charging_data['available_capacity'] = None

    # 初始化变量
    total_current_integral = 0
    soc_initial = None
    start_time = None  # 初始化起始时间

    # 检查第一行是否是充电状态
    if charging_data['charge_state'].iloc[0] == 1:  # 充电状态是1时，初始化
        total_current_integral = 0
        soc_initial = charging_data['bat_user_soc_hvs'].iloc[0]
        start_time = charging_data['datatime'].iloc[0]

    # 遍历充电数据的行
    for i in range(len(charging_data)):
        # 如果是充电状态从非充电状态变成1的行，重新初始化
        if i > 0 and charging_data['charge_state'].iloc[i - 1] not in [1] and charging_data['charge_state'].iloc[i] == 1:
            total_current_integral = 0
            soc_initial = charging_data['bat_user_soc_hvs'].iloc[i]
            start_time = charging_data['datatime'].iloc[i]  # 重新初始化起始时间

        # 当前时间间隔（单位：秒）
        if i > 0 and charging_data['charge_state'].iloc[i] == 1 and charging_data['charge_state'].iloc[i - 1] == 1:
            time_diff = (charging_data['datatime'].iloc[i] - start_time).total_seconds()
            # 当前电流积分（梯形法）
            current_integral = (charging_data['bat_i_hvs'].iloc[i] + charging_data['bat_i_hvs'].iloc[i - 1]) / 2 * time_diff
            # 累积电流积分
            total_current_integral += current_integral
            # 更新起始时间为当前时间
            start_time = charging_data['datatime'].iloc[i]

        # 当前SOC变化
        if soc_initial is not None and charging_data['charge_state'].iloc[i] == 1:
            soc_final = charging_data['bat_user_soc_hvs'].iloc[i]
            soc_change = soc_final - soc_initial

            # 计算当前可用容量（去掉负号）
            if soc_change != 0:
                Ca = (total_current_integral / soc_change) / 1000 * 100 / 3600  # 移除负号
                charging_data.at[i, 'available_capacity'] = Ca

    # 去除charging_data中的重复行
    charging_data = charging_data.drop_duplicates(subset='datatime', keep='first')

    # 将可用容量列合并回原始数据
    data = pd.merge(data, charging_data[['datatime', 'available_capacity']], on='datatime', how='left')

    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)

    # 构建新的文件路径
    new_file_path = os.path.join(output_folder, file_name)

    # 保存新的CSV文件
    data.to_csv(new_file_path, index=False)

    print(f"新的CSV文件已保存到: {new_file_path}")



import subprocess
subprocess.run(['python', r"F:\妙算榜-赛题信息\CATL数据处理\33提取_里程-容量.py"])
