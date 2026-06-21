import pandas as pd
import os
import glob
import shutil
from datetime import timedelta

# 第一部分参数
input_dir = r'F:\妙算榜-赛题信息\CATL_去除异常值(SOC和充电状态)'
output_dir = r'F:\妙算榜-赛题信息\CATL_充电指定SOC区间'
min_soc = 50
max_soc = 80

# 第一部分：处理充电指定SOC区间的数据
def process_soc_range(input_dir, output_dir, min_soc, max_soc):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有CSV文件
    file_list = glob.glob(os.path.join(input_dir, '*.csv'))

    # 遍历所有符合条件的 CSV 文件
    for file_path in file_list:
        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 备份当前运行的Python文件
        current_py_file = os.path.basename(__file__)
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
                elif is_in_charging_range and soc == max_soc and not found_soc_max:  # 找到SOC为80的第一行
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

# 第二部分：处理积分容量并生成里程-容量数据
def process_integral_capacity(source_folder, output_folder):
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有CSV文件
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

if __name__ == "__main__":


    # 执行第一部分处理
    process_soc_range(input_dir, output_dir, min_soc, max_soc)

    # 第二部分参数
    source_folder = r"F:\妙算榜-赛题信息\CATL_充电指定SOC区间"
    output_folder = r'F:\妙算榜-赛题信息\CATL_积分容量'

    # 执行第二部分处理
    process_integral_capacity(source_folder, output_folder)

# 运行额外的Python脚本
subprocess.run(['python', r"F:\妙算榜-赛题信息\CATL数据处理\33提取_里程-容量.py"])