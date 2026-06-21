import os
import pandas as pd

# 定义输入和输出目录
input_dir = r'F:\妙算榜-赛题信息\CATL_时间顺序'
output_dir = r'F:\妙算榜-赛题信息\CATL_去除异常值(SOC和充电状态)'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有 CSV 文件
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        # 读取 CSV 文件
        file_path = os.path.join(input_dir, file_name)
        data = pd.read_csv(file_path)

        # 删除 charge_state > 5 或 bat_user_soc_hvs > 100 的行
        data = data[(data['charge_state'] <= 5) & (data['bat_user_soc_hvs'] <= 100)]

        # 保存处理后的文件到输出目录
        output_file_path = os.path.join(output_dir, file_name)
        data.to_csv(output_file_path, index=False)

        print(f'已处理并保存文件: {file_name}')

print('所有文件处理完成！')