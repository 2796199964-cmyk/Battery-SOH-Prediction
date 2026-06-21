import os
import pandas as pd

# 定义源目录和目标目录
source_dir = r'F:\妙算榜-赛题信息\车辆使用寿命预测'
target_dir = r'F:\妙算榜-赛题信息\CATL_精简后数据'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 需要保留的列（去掉前缀后的列名）
required_columns = [
    'datatime', 'vehicle_state', 'charge_state', 'run_state',
    'speed', 'odometer', 'bat_out_hvs', 'bat_i_hvs', 'bat_user_soc_hvs'
]

# 遍历源目录中的所有CSV文件
for file_name in os.listdir(source_dir):
    if file_name.endswith('.csv'):
        # 读取CSV文件，指定编码为gbk
        file_path = os.path.join(source_dir, file_name)
        try:
            df = pd.read_csv(file_path, encoding='gbk')  # 尝试使用gbk编码
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin1')  # 如果gbk失败，尝试latin1
            except Exception as e:
                print(f"文件 {file_name} 读取失败，错误: {e}")
                continue

        # 去掉表头中的前缀
        df.columns = [col.replace('sdp_electric_vehicles_data_chj.', '') for col in df.columns]

        # 只保留需要的列
        df_filtered = df[required_columns]

        # 保存处理后的文件到目标目录
        output_file_path = os.path.join(target_dir, file_name)
        df_filtered.to_csv(output_file_path, index=False, encoding='utf-8')  # 保存为UTF-8编码

        print(f"文件 {file_name} 处理完成，已保存到 {output_file_path}")

print("所有文件处理完成！")