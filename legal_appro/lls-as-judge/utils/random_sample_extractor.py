import json
import random
import os
from pathlib import Path

# 设置随机种子以确保结果可重现
random.seed(42)

# 定义源文件夹路径
source_folders = [
    {
        'path': r'e:\project_4\CaseGen-main\CaseGen-main\eval\llm_eval_r1_711_cmdl',
        'tag': 'cmdl'
    },
    {
        'path': r'e:\project_4\CaseGen-main\CaseGen-main\eval\llm_eval_r1_711_Judge',
        'tag': 'judge'
    },
    {
        'path': r'e:\project_4\CaseGen-main\CaseGen-main\eval\llm_eval_r1_711_SLJA',
        'tag': 'SLJA'
    }
]

# 定义总样本数量和源文件夹数量
total_samples = 50
num_source_folders = len(source_folders)  # 应该是3个源文件夹

# 计算每个文件平均抽取的样本数量
samples_per_file = total_samples // num_source_folders
remaining_samples = total_samples % num_source_folders

# 存储所有抽取的样本
all_samples = []

# 遍历每个源文件夹
for source in source_folders:
    source_path = Path(source['path'])
    source_tag = source['tag']
    
    # 获取源文件夹下的所有子文件夹
    subfolders = [f for f in source_path.iterdir() if f.is_dir()]
    
    # 遍历每个子文件夹，只处理带有"gemini_"前缀的子文件夹
    for subfolder in subfolders:
        if not subfolder.name.startswith('gemini_'):
            continue
        # 构建criminal_new.json文件的路径
        file_path = subfolder / 'criminal_new.json'
        
        # 检查文件是否存在
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            continue
        
        # 读取文件内容（JSON Lines格式）
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"解析文件 {file_path} 第 {line_num} 行时出错: {e}")
                        # 继续处理其他行
        except UnicodeDecodeError as e:
            print(f"读取文件 {file_path} 时编码错误: {e}")
            continue
        
        if not data:
            print(f"文件 {file_path} 中没有有效数据")
            continue
        
        # 检查数据数量是否足够
        if len(data) < samples_per_file:
            print(f"文件 {file_path} 中的数据不足 {samples_per_file} 条，实际有 {len(data)} 条")
            continue
        
        # 检查数据数量是否足够
        available_samples = min(len(data), samples_per_file)
        if available_samples < samples_per_file:
            print(f"文件 {file_path} 中的数据不足 {samples_per_file} 条，实际有 {len(data)} 条")
            
        # 随机抽取样本
        sampled_data = random.sample(data, available_samples)
        
        # 为每个样本添加来源标记
        for sample in sampled_data:
            sample['source'] = source_tag
            sample['model'] = subfolder.name.replace(f'_{source_tag}', '')  # 提取模型名称
        
        # 将抽取的样本添加到总列表
        all_samples.extend(sampled_data)
        
        print(f"从 {file_path} 抽取了 {len(sampled_data)} 个样本")

# 分配剩余样本
if remaining_samples > 0:
    print(f"\n分配剩余的 {remaining_samples} 个样本...")
    
    # 遍历源文件夹，为每个源文件夹分配剩余样本
    for source in source_folders:
        if remaining_samples <= 0:
            break
            
        source_path = Path(source['path'])
        source_tag = source['tag']
        
        # 找到当前源文件夹下的gemini子文件夹
        gemini_subfolders = [f for f in source_path.iterdir() if f.is_dir() and f.name.startswith('gemini_')]
        if gemini_subfolders:
            # 获取gemini子文件夹路径
            gemini_subfolder = gemini_subfolders[0]
            file_path = gemini_subfolder / 'criminal_new.json'
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue  # 跳过空行
                        try:
                            json_obj = json.loads(line)
                            data.append(json_obj)
                        except json.JSONDecodeError as e:
                            continue  # 跳过解析错误的行
            except:
                data = []
            
            if data:
                # 确保不会重复抽取已抽取的样本
                existing_ids = [sample.get('id') for sample in all_samples if 'id' in sample]
                available_data = [item for item in data if item.get('id') not in existing_ids]
                
                if available_data:
                    # 抽取一个额外样本
                    extra_sample = random.sample(available_data, 1)
                    # 添加来源标记
                    for sample in extra_sample:
                        sample['source'] = source_tag
                        sample['model'] = gemini_subfolder.name.replace(f'_{source_tag}', '')
                    # 添加到总列表
                    all_samples.extend(extra_sample)
                    remaining_samples -= 1
                    print(f"从 {file_path} 额外抽取了 1 个样本")

# 将所有抽取的样本保存到新的JSON文件
output_file = Path(r'e:\project_4\CaseGen-main\CaseGen-main\random_samples.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

print(f"\n所有样本已保存到 {output_file}")
print(f"总共抽取了 {len(all_samples)} 个样本")
