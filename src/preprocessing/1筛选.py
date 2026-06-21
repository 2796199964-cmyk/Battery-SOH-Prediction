import os
import pandas as pd
from glob import glob

# 定义源文件夹
source_folder = r'F:\妙算榜-赛题信息\车辆使用寿命预测'

# 获取所有符合条件的CSV文件路径
csv_files = glob(os.path.join(source_folder, '0001' + '*.csv'))

# 对于每个CSV文件执行数据处理
for file_path in csv_files:
    # 加载数据
    df = pd.read_csv(file_path)

    # 检查是否存在目标列并进行排序
    if 'sdp_electric_vehicles_data_chj.datatime' in df.columns and 'sdp_electric_vehicles_data_chj.bat_user_soc_hvs' in df.columns:
        df_sorted = df.sort_values(by='sdp_electric_vehicles_data_chj.datatime', ascending=True)

        # 筛选递增的数据
        col_name = 'sdp_electric_vehicles_data_chj.bat_user_soc_hvs'
        mask = df_sorted[col_name].diff().fillna(0) >= 0  # 保留递增（包括相等）的数据
        df_filtered = df_sorted[mask]

        # 创建新的文件路径
        new_file_path = os.path.splitext(file_path)[0] + '_1' + os.path.splitext(file_path)[1]

        # 保存处理后的数据到新文件
        df_filtered.to_csv(new_file_path, index=False)
        print(f"已处理并保存文件至 {new_file_path}")
    else:
        if 'sdp_electric_vehicles_data_chj.datatime' not in df.columns:
            print(f"警告: 文件 {file_path} 中不包含列 sdp_electric_vehicles_data_chj.datatime，跳过此文件。")
        if 'sdp_electric_vehicles_data_chj.bat_user_soc_hvs' not in df.columns:
            print(f"警告: 文件 {file_path} 中不包含列 sdp_electric_vehicles_data_chj.bat_user_soc_hvs，跳过此文件。")