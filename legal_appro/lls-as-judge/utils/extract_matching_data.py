import json
import os
from pathlib import Path

def main():
    # 定义文件路径
    random_samples_path = Path(r'e:\project_4\CaseGen-main\CaseGen-main\random_samples.json')
    prompt_folders = [
        Path(r'generate\generated_data\gemini_cmdl'),
        Path(r'generate\generated_data\gemini_judge'),
        Path(r'generate\generated_data\gemini_SLJA')
    ]
    output_path = Path(r'e:\project_4\CaseGen-main\CaseGen-main\matching_data.json')
    
    try:
        # 读取random_samples.json文件并筛选出满足条件的数据
        with open(random_samples_path, 'r', encoding='utf-8') as f:
            random_samples = json.load(f)
        
        print(f"成功读取random_samples.json，共包含 {len(random_samples)} 个样本")
        
        # 筛选出满足条件的数据项 (model == 'gemini' and source in ['cmdl', 'Judge', 'SLJA'])
        filtered_samples = [sample for sample in random_samples 
                          if (sample.get('source') == 'cmdl' or sample.get('source') == 'judge' or sample.get('source') == 'SLJA') and sample.get('model') == 'gemini']
        
        print(f"筛选出满足条件的样本数量: {len(filtered_samples)}")
        
        # 按source类型分组id，提高匹配准确性
        ids_by_source = {
            'cmdl': set(),
            'judge': set(),
            'SLJA': set()
        }
        for sample in filtered_samples:
            source = sample.get('source')
            if source in ids_by_source:
                ids_by_source[source].add(sample['id'])
        
        # 显示各source的id数量
        for source, ids in ids_by_source.items():
            print(f"{source} 类型的id数量: {len(ids)}")
        
        # 收集匹配的数据
        matching_data = []
        
        # 定义source到文件夹的映射关系
        source_to_folder = {
            'cmdl': prompt_folders[0],  # gemini_cmdl
            'judge': prompt_folders[1],  # gemini_judge
            'SLJA': prompt_folders[2]    # gemini_SLJA
        }
        
        # 遍历每个source，在对应的文件夹中查找匹配的id
        for source, target_ids in ids_by_source.items():
            if not target_ids:
                print(f"\n{source} 类型没有需要匹配的id")
                continue
                
            folder = source_to_folder[source]
            print(f"\n处理 {source} 类型的数据，在文件夹: {folder} 中查找")
            
            # 查找criminal_new_eval_prompt.json文件
            prompt_file = folder / 'criminal_new.json'
            if not prompt_file.exists():
                print(f"文件不存在: {prompt_file}")
                continue
            
            try:
                # 读取prompt文件（JSON Lines格式）
                prompt_data = []
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    # for line_num, line in enumerate(f, 1):
                        prompt_data=  json.load(f)
                        # line = line.strip()
                        # if not line:
                        #     continue  # 跳过空行
                        # try:
                        #     json_obj = json.loads(line)
                        #     prompt_data.append(json_obj)
                        # except json.JSONDecodeError as e:
                        #     print(f"解析文件 {prompt_file} 第 {line_num} 行时出错: {e}")
                            # 继续处理其他行
                
                print(f"成功读取 {prompt_file}，共包含 {len(prompt_data)} 个有效数据项")
                
                # 查找匹配的id
                match_count = 0
                for item in prompt_data:
                    if isinstance(item, dict) and 'id' in item and item['id'] in target_ids:
                        # 添加匹配的数据项，包含来源信息
                        item['prompt_source'] = folder.name
                        item['sample_source'] = source  # 添加原始样本的source类型
                        matching_data.append(item)
                        print(f"找到匹配的id: {item['id']}")
                        
                        # 从目标id集合中移除已找到的id，避免重复匹配
                        target_ids.remove(item['id'])
                        match_count += 1
                        
                print(f"从 {prompt_file} 中找到 {match_count} 个匹配项")
                
                # 检查该source是否有未匹配到的id
                if target_ids:
                    print(f"警告: {source} 类型中有以下id未找到匹配项: {sorted(target_ids)}")
                
            except json.JSONDecodeError as e:
                print(f"解析文件 {prompt_file} 时出错: {e}")
            except Exception as e:
                print(f"处理文件 {prompt_file} 时出错: {e}")
        
        # 检查是否有未匹配到的id
        if target_ids:
            print(f"\n警告: 以下id未在prompt文件中找到匹配项: {sorted(target_ids)}")
        
        # 保存匹配的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matching_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！")
        print(f"总共找到 {len(matching_data)} 个匹配的数据项")
        print(f"匹配的数据已保存到: {output_path}")
        
    except json.JSONDecodeError as e:
        print(f"解析random_samples.json文件时出错: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()