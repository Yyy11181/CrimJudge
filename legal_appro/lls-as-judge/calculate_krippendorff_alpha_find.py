import pandas as pd
import numpy as np
import os
from glob import glob
import itertools

# 收集所有打分文件
def collect_all_score_files():
    score_files = glob("lls-as-judge/人工打分/*.xlsx")
    print(f"找到的打分文件数量: {len(score_files)}")
    print("文件列表:")
    for file in score_files:
        print(f"  - {file}")
    return score_files

# 检查所有文件的结构是否一致
# 检查所有文件是否包含必要的评分列
def check_files_consistency(score_files):
    print("\n检查文件结构一致性...")
    # 必要的评分列
    required_columns = ['事实准确性', '法律逻辑性', '量刑情节完备性', '语言与格式规范性']
    all_columns_consistent = True
    
    # 检查文件行数
    all_row_counts = []
    for file in score_files:
        df = pd.read_excel(file)
        all_row_counts.append(len(df))
    
    # 检查所有文件的行数是否一致
    rows_consistent = all(count == all_row_counts[0] for count in all_row_counts)
    print(f"行数一致性: {'一致' if rows_consistent else '不一致'}")
    print(f"各文件行数: {all_row_counts}")
    
    # 检查每个文件是否包含必要的评分列
    for i, file in enumerate(score_files):
        df = pd.read_excel(file)
        has_required = all(col in df.columns for col in required_columns)
        print(f"文件 {i+1} ({os.path.basename(file)}) 包含所有必要评分列: {'是' if has_required else '否'}")
        if not has_required:
            all_columns_consistent = False
    
    print(f"所有文件都包含必要评分列: {'是' if all_columns_consistent else '否'}")
    
    return all_columns_consistent, rows_consistent

# 提取评分数据
def extract_scores(score_files):
    print("\n提取评分数据...")
    score_data = {}
    
    # 需要提取的评分列
    score_columns = ['事实准确性', '法律逻辑性', '量刑情节完备性', '语言与格式规范性']
    
    for file in score_files:
        df = pd.read_excel(file)
        # 使用id作为索引
        df = df.set_index('id')
        # 只保留评分列
        scores = df[score_columns]
        # 获取评分者姓名（从文件名提取）
        rater_name = os.path.basename(file).split('.')[0]
        score_data[rater_name] = scores
    
    return score_data

# 确保所有评分者的数据对齐
def align_scores(score_data):
    print("\n对齐评分数据...")
    # 获取所有评分者共有的id
    all_ids = set.intersection(*[set(df.index) for df in score_data.values()])
    # 将集合转换为列表
    all_ids = sorted(list(all_ids))
    print(f"共有 {len(all_ids)} 个共同的样本ID")
    
    # 对齐数据
    aligned_data = {}
    for rater, df in score_data.items():
        aligned_data[rater] = df.loc[all_ids]
    
    return aligned_data

# 实现Krippendorff α计算
def krippendorff_alpha(data_matrix):
    """
    计算Krippendorff α系数
    data_matrix: 二维数组，行是样本，列是评分者，值是评分
    """
    # 将数据转换为numpy数组
    data = np.array(data_matrix)
    
    # 获取样本数和评分者数
    n_samples, n_raters = data.shape
    
    # 计算每个样本的有效评分者数
    n_effective = np.sum(~np.isnan(data), axis=1)
    
    # 计算观察值之间的差异
    observed_differences = 0
    total_observations = 0
    
    for i in range(n_samples):
        # 获取第i个样本的所有评分（去除NaN）
        sample_ratings = data[i, ~np.isnan(data[i, :])]
        k = len(sample_ratings)
        
        if k < 2:
            continue
        
        # 计算该样本的观察差异
        for j in range(k):
            for l in range(j+1, k):
                observed_differences += (sample_ratings[j] - sample_ratings[l]) ** 2
        
        total_observations += k * (k - 1)
    
    if total_observations == 0:
        return np.nan
    
    # 计算观察差异的平均值
    observed_agreement = observed_differences / total_observations
    
    # 计算期望差异（假设随机评分）
    # 获取所有评分值
    all_ratings = data[~np.isnan(data)]
    
    if len(all_ratings) == 0:
        return np.nan
    
    # 计算评分分布
    values, counts = np.unique(all_ratings, return_counts=True)
    probabilities = counts / len(all_ratings)
    
    expected_differences = 0
    for i in range(len(values)):
        for j in range(len(values)):
            expected_differences += probabilities[i] * probabilities[j] * (values[i] - values[j]) ** 2
    
    # 计算Krippendorff α
    if expected_differences == 0:
        return 1.0
    
    alpha = 1 - (observed_agreement / expected_differences)
    return alpha

# 计算给定文件组合的Krippendorff α系数
def calculate_alpha_for_combination(file_combination):
    """
    计算给定文件组合的Krippendorff α系数
    """
    # 不输出处理组合的详细信息，只返回结果
    # 检查文件一致性
    columns_consistent, rows_consistent = check_files_consistency(file_combination)
    
    if not columns_consistent or not rows_consistent:
        return None, None
    
    # 提取评分数据
    score_data = extract_scores(file_combination)
    
    # 对齐评分数据
    aligned_data = align_scores(score_data)
    
    # 计算每个维度的Krippendorff α
    score_columns = ['事实准确性', '法律逻辑性', '量刑情节完备性', '语言与格式规范性']
    dimension_alphas = {}
    
    for column in score_columns:
        # 构建该维度的评分矩阵
        matrix = []
        for rater, df in aligned_data.items():
            matrix.append(df[column].tolist())
        
        # 转置矩阵，使行是样本，列是评分者
        matrix = np.array(matrix).T
        
        # 计算Krippendorff α
        alpha = krippendorff_alpha(matrix)
        dimension_alphas[column] = alpha
    
    # 计算整体的Krippendorff α
    all_ratings = []
    for rater, df in aligned_data.items():
        for column in score_columns:
            all_ratings.append(df[column].tolist())
    
    # 转置矩阵
    all_ratings = np.array(all_ratings).T
    overall_alpha = krippendorff_alpha(all_ratings)
    
    return overall_alpha, dimension_alphas

# 主函数
def main():
    print("计算Krippendorff α系数")
    print("="*50)
    
    # 收集所有打分文件
    all_score_files = collect_all_score_files()
    
    if len(all_score_files) < 2:
        print("\n错误：需要至少2个打分文件才能进行组合分析")
        return
    
    # 生成所有可能大小的组合（从2个到所有文件）
    combinations = []
    for r in range(2, len(all_score_files) + 1):
        combinations.extend(list(itertools.combinations(all_score_files, r)))
    
    print(f"\n\n生成所有可能大小的组合，共 {len(combinations)} 个组合")
    print(f"组合大小范围：2到{len(all_score_files)}个文件")
    print("\n开始计算所有组合的α系数...")
    
    # 记录所有组合的结果
    results = []
    
    # 处理每个组合
    for i, combination in enumerate(combinations, 1):
        # 只输出进度信息
        if i % 10 == 0 or i == len(combinations):
            print(f"处理第 {i}/{len(combinations)} 个组合...")
        
        overall_alpha, dimension_alphas = calculate_alpha_for_combination(combination)
        
        if overall_alpha is not None:
            # 保存结果
            results.append({
                'combination': combination,
                'size': len(combination),
                'overall_alpha': overall_alpha,
                'dimension_alphas': dimension_alphas
            })
    
    # 找出整体α系数最高的组合
    if results:
        best_result = max(results, key=lambda x: x['overall_alpha'])
        
        print(f"\n\n{'='*70}")
        print("最佳组合结果")
        print(f"{'='*70}")
        print(f"最佳组合大小: {best_result['size']}个文件")
        print(f"最佳组合的文件: {[os.path.basename(f) for f in best_result['combination']]}")
        print(f"整体Krippendorff α系数: {best_result['overall_alpha']:.4f}")
        print("各维度Krippendorff α系数:")
        for column, alpha in best_result['dimension_alphas'].items():
            print(f"  {column}: {alpha:.4f}")
    else:
        print("\n\n没有找到有效的文件组合")

if __name__ == "__main__":
    main()