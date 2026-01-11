import json
import pandas as pd
import re

# 读取JSON文件
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # data = json.load(f)
            data = [json.loads(line) for line in f]
        return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None

# 从单个条目提取评分
def extract_scores_from_entry(entry):
    scores = {
        'id': entry.get('id', None),
        'source': entry.get('source', None),
        'model': entry.get('model', None),
        '事实准确性': None,
        '法律逻辑性': None,
        '量刑情节完备性': None,
        '语言与格式规范性': None
    }
    
    # 获取response字段
    response = entry.get('response', '')
    
    # 检查response是否为空
    if not response:
        print(f"解析评分时出错 (id={scores['id']}): response为空")
        return scores
    
    try:
        # 尝试直接解析JSON格式的response
        if isinstance(response, str):
            # 处理可能包含代码块的情况
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            else:
                json_str = response.strip()
            
            # 检查json_str是否为空
            if not json_str:
                print(f"解析评分时出错 (id={scores['id']}): 提取的JSON字符串为空")
                raise ValueError("JSON字符串为空")
            
            # 解析JSON
            response_data = json.loads(json_str)
            if 'scores' in response_data:
                scores_data = response_data['scores']
                scores['事实准确性'] = scores_data.get('事实准确性', None)
                scores['法律逻辑性'] = scores_data.get('法律逻辑性', None)
                scores['量刑情节完备性'] = scores_data.get('量刑情节完备性', None)
                scores['语言与格式规范性'] = scores_data.get('语言与格式规范性', None)
    except Exception as e:
        print(f"解析评分时出错 (id={scores['id']}): {e}")
        # 尝试使用正则表达式提取分数
        try:
            # 定义正则表达式模式
            patterns = {
                '事实准确性': r'事实准确性.*?:\s*([0-9])',
                '法律逻辑性': r'法律逻辑性.*?:\s*([0-9])',
                '量刑情节完备性': r'量刑情节完备性.*?:\s*([0-9])',
                '语言与格式规范性': r'语言与格式规范性.*?:\s*([0-9])'
            }
            
            # 提取各维度分数
            for key, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    scores[key] = int(match.group(1))
        except Exception as e2:
            print(f"正则表达式提取分数时出错 (id={scores['id']}): {e2}")
    
    return scores

# 主函数
def main():
    # 定义文件路径
    json_file_path = r'eval\llm_eval_r1_all\all\criminal_new_7.json'
    output_excel_path = r'lls-as-judge\llm打分\llm8.xlsx'
    
    # 读取JSON文件
    data = read_json_file(json_file_path)
    if not data:
        return
    
    # 确保data是列表类型
    if not isinstance(data, list):
        print("JSON数据格式不正确，应为列表类型")
        return
    
    # 提取所有评分
    scores_list = []
    for entry in data:
        scores = extract_scores_from_entry(entry)
        scores_list.append(scores)
    
    # 创建DataFrame
    df = pd.DataFrame(scores_list)
    
    # 调整列顺序
    columns_order = ['id', 'source', 'model', '事实准确性', '法律逻辑性', '量刑情节完备性', '语言与格式规范性']
    df = df[columns_order]
    
    # 保存到Excel文件
    try:
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"评分数据已成功提取并保存到: {output_excel_path}")
        print(f"共处理 {len(df)} 条数据")
        print(f"成功提取评分的数据: {len(df.dropna(subset=['事实准确性', '法律逻辑性', '量刑情节完备性', '语言与格式规范性']))} 条")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

if __name__ == "__main__":
    main()