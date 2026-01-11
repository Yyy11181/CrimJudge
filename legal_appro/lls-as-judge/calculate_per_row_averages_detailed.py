import pandas as pd
import numpy as np
import re
import os
import glob

def extract_score(score_str):
    """
    从字符串中提取分数，支持以下格式：
    - 纯数字："5"
    - 数字加括号："5（非常好）"
    - 数字加逗号："5,优秀"
    """
    if pd.isna(score_str):
        return None
    
    # 转换为字符串
    score_str = str(score_str)
    
    # 使用正则表达式提取数字部分
    match = re.search(r'\d+(\.\d+)?', score_str)
    if match:
        return float(match.group())
    return None

def calculate_per_row_averages_detailed():
    """
    计算三个文件中每行数据的平均分，并显示详细的计算过程
    """
    # 遍历lls-as-judge/裁判文书打分文件夹中的所有xlsx文件
    files = glob.glob('lls-as-judge/人工打分/*.xlsx')
    
    # 读取所有文件
    dfs = []
    file_names = []
    for file in files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                dfs.append(df)
                file_names.append(os.path.basename(file))
                print(f"成功读取文件: {file}，包含 {len(df)} 行数据")
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        else:
            print(f"文件不存在: {file}")
    
    # if len(dfs) < 3:
    #     print("无法继续，需要三个文件都存在")
    #     return
    
    # 收集所有id
    all_ids = set()
    for df in dfs:
        if 'id' in df.columns:
            all_ids.update(df['id'].dropna().astype(int))
    
    print(f"\n所有文件中共有 {len(all_ids)} 个不同的id")
    
    # 定义维度映射
    dimension_mapping = {
        '事实准确性': [ '事实准确性'],
        '法律逻辑性': [ '法律逻辑性'],
        '量刑情节完备性': ['量刑情节完备性'],
        '语言与格式规范性': ['语言与格式规范性']
    }
    
    # 找到每个文件实际使用的列名
    file_columns = []
    for df in dfs:
        actual_columns = {}
        for dim, possible_names in dimension_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    actual_columns[dim] = name
                    break
        file_columns.append(actual_columns)
    
    # 创建结果列表
    results = []
    
    # 为前5个id显示详细计算过程
    show_detailed = 5
    count = 0
    
    # 为每个id计算平均分
    for id_value in sorted(all_ids):
        # 收集该id在所有文件中的分数
        id_scores = {dim: [] for dim in dimension_mapping.keys()}
        file_raw_scores = {dim: [] for dim in dimension_mapping.keys()}
        
        for i, (df, columns, file_name) in enumerate(zip(dfs, file_columns, file_names)):
            if 'id' in df.columns:
                # 查找该id的行
                id_rows = df[df['id'] == id_value]
                if len(id_rows) > 0:
                    row = id_rows.iloc[0]
                    
                    # 提取每个维度的分数
                    for dim, col in columns.items():
                        if col in row:
                            raw_score = row[col]
                            score = extract_score(raw_score)
                            if score is not None:
                                id_scores[dim].append(score)
                                file_raw_scores[dim].append((file_name, raw_score, score))
        
        # 计算每个维度的平均分
        avg_scores = {dim: round(np.mean(scores), 2) if len(scores) >= 1 else None for dim, scores in id_scores.items()}
        
        # 添加到结果中
        results.append({
            'id': id_value,
            **avg_scores
        })
        
        # 显示详细计算过程
        if count < show_detailed:
            count += 1
            print(f"\n=== ID {id_value} 的详细计算过程 ===")
            for dim in dimension_mapping.keys():
                if file_raw_scores[dim]:
                    print(f"\n{dim}:")
                    total = 0
                    for file_name, raw_score, score in file_raw_scores[dim]:
                        print(f"  {file_name}: {raw_score} -> 提取分数: {score}")
                        total += score
                    avg = total / len(file_raw_scores[dim])
                    print(f"  总分数: {total}, 平均分: {avg:.4f}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 显示结果
    print("\n处理完成！计算结果：")
    print(f"共处理 {len(result_df)} 个样本")
    
    # 显示前10个样本
    print("\n前10个样本的平均分:")
    print(result_df.head(10))
    
    # 保存结果到Excel文件
    output_file = 'lls-as-judge/human-new.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    # 计算并显示总体平均分
    print("\n各维度总体平均分:")
    for dim in dimension_mapping.keys():
        if dim in result_df.columns:
            avg = result_df[dim].mean()
            print(f"{dim}: {round(avg)}")

if __name__ == "__main__":
    calculate_per_row_averages_detailed()