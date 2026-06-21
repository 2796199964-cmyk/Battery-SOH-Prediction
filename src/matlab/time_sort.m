% 定义源目录和目标目录
source_dir = 'F:\妙算榜-赛题信息\车辆使用寿命预测\';
target_dir = 'F:\妙算榜-赛题信息\车辆使用寿命预测\CATL_精简后数据';

% 确保目标目录存在
if ~exist(target_dir, 'dir')
    mkdir(target_dir);
end

% 获取源目录中所有的CSV文件
file_list = dir(fullfile(source_dir, '*.csv'));

% 遍历每个文件
for i = 1:length(file_list)
    % 读取CSV文件
    file_name = file_list(i).name;
    file_path = fullfile(source_dir, file_name);
    data = readtable(file_path, 'Delimiter', ',', 'TextType', 'string');
    
    % 将datatime列转换为datetime类型
    data.datatime = datetime(data.datatime, 'InputFormat', 'yyyy/MM/dd HH:mm:ss');
    
    % 按照datatime列升序排序
    data_sorted = sortrows(data, 'datatime');
    
    % 保存排序后的文件到目标目录
    output_file_path = fullfile(target_dir, file_name);
    writetable(data_sorted, output_file_path, 'Delimiter', ',');
    
    fprintf('文件 %s 处理完成，已保存到 %s\n', file_name, output_file_path);
end

disp('所有文件处理完成！');