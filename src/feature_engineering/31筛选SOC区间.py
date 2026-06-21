import pandas as pd
import os
import glob
import shutil

# 定义目录路径
input_dir = r'F:\妙算榜-赛题信息\CATL_去除异常值(SOC和充电状态)'
output_dir = r'F:\妙算榜-赛题信息\CATL_充电指定SOC区间'

min_soc = 50
max_soc = 80

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取当前运行的Python文件名
current_py_file = os.path.basename(__file__)

# 获取所有以 '0001' 开头的 CSV 文件
file_list = glob.glob(os.path.join(input_dir, '0001*.csv'))

# 遍历所有符合条件的 CSV 文件
for file_path in file_list:
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 备份当前运行的Python文件
    backup_py_file_name = f"{os.path.splitext(file_path)[0]}_区间[{min_soc}-{max_soc}]_{current_py_file}"
    backup_py_file_path = os.path.join(output_dir, backup_py_file_name)
    shutil.copyfile(__file__, backup_py_file_path)
    print(f"当前py文件已备份到: {backup_py_file_path}")

    # 转换时间戳列为datetime类型
    data['datatime'] = pd.to_datetime(data['datatime'])

    # 按时间戳排序
    data = data.sort_values('datatime').reset_index(drop=True)

    # 标记筛选后的数据
    filtered_data = []

    # 初始化SOC的标记
    is_in_charging_range = False
    found_soc_max = False  # 用于标记是否已经找到SOC为80的第一行

    # 遍历数据行，筛选充电状态为1时SOC在50到80之间的数据
    for i in range(len(data)):
        if data['charge_state'].iloc[i] == 1:  # 如果是充电状态1
            soc = data['bat_user_soc_hvs'].iloc[i]

            # 判断是否处于SOC从50上升到80的区间
            if not is_in_charging_range and soc >= min_soc:  # SOC第一次达到50
                is_in_charging_range = True
                found_soc_max = False  # 重置SOC为80的标记

            if is_in_charging_range and soc < max_soc:  # SOC仍在50到80之间
                filtered_data.append(data.iloc[i])
            elif is_in_charging_range and soc == max and not found_soc_80:  # 找到SOC为80的第一行
                filtered_data.append(data.iloc[i])
                found_soc_max = True  # 标记已经找到SOC为80的第一行
            elif is_in_charging_range and soc > max_soc:  # SOC超过80，结束这个充电阶段
                is_in_charging_range = False
        else:
            # 充电状态为2、3、4的数据保持不变
            filtered_data.append(data.iloc[i])

    # 将筛选后的数据转为DataFrame
    filtered_data = pd.DataFrame(filtered_data)

    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)

    # 构建新的文件路径
    new_file_path = os.path.join(output_dir, file_name)

    # 保存筛选后的数据到新的CSV文件
    filtered_data.to_csv(new_file_path, index=False)

    print(f"筛选后的新文件已保存到: {new_file_path}")

import subprocess
subprocess.run(['python', r"F:\妙算榜-赛题信息\CATL数据处理\32电池容量积分.py"])
