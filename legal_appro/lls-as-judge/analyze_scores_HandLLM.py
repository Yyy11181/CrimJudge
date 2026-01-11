import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import krippendorff
import sys

# --- 1. 配置区域 ---
LLM_SCORES_FILE = r'lls-as-judge\llm-new.xlsx'
HUMAN_SCORES_FILE = r'lls-as-judge\human-new.xlsx'
MERGE_COLUMN_ID = 'id'

DIMENSIONS_TO_ANALYZE = {
    '事实准确性': ('事实准确性', '事实准确性'),
    '法律逻辑性': ('法律逻辑性', '法律逻辑性'),
    '量刑情节完备性': ('量刑情节完备性', '量刑情节完备性'),
    '语言与格式规范性': ('语言与格式规范性', '语言与格式规范性')
}

def analyze_agreement(llm_filepath, human_filepath, id_col, dimensions):

    try:
        llm_df = pd.read_excel(llm_filepath)
        human_df = pd.read_excel(human_filepath)
    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e}")
        sys.exit(1)

    merged_df = pd.merge(
        llm_df.drop_duplicates(id_col),
        human_df.drop_duplicates(id_col),
        on=id_col,
        suffixes=('_llm', '_human')
    )

    if merged_df.empty:
        print("[错误] 合并后数据为空")
        return

    results = []

    for display_name, (llm_col_base, human_col_base) in dimensions.items():
        llm_col = f'{llm_col_base}_llm'
        human_col = f'{human_col_base}_human'

        print(f"\n--- 正在分析维度: {display_name} ---")

        temp_df = merged_df[[llm_col, human_col]].dropna()

        if temp_df.empty:
            print("  [信息] 无有效数据")
            continue

        llm_scores = temp_df[llm_col].astype(int).to_numpy()
        human_scores = temp_df[human_col].astype(int).to_numpy()

        llm_mean = llm_scores.mean()
        human_mean = human_scores.mean()

        # # Spearman
        # rho, rho_p = spearmanr(human_scores, llm_scores)

        # Kendall τ-b
        tau, tau_p = kendalltau(human_scores, llm_scores)

        # Krippendorff’s α (ordinal)
        try:
            reliability_data = np.vstack([llm_scores, human_scores])
            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="ordinal"
            )
        except Exception as e:
            print(f"  [警告] Krippendorff α 计算失败: {e}")
            alpha = np.nan

        results.append({
            '维度': display_name,
            '有效样本数': len(llm_scores),
            'LLM均分': f"{llm_mean:.2f}",
            '人工均分': f"{human_mean:.2f}",
            # 'Spearman ρ': f"{rho:.4f}",
            # 'ρ p值': f"{rho_p:.4f}",
            'Kendall τ': f"{tau:.4f}",
            'τ p值': f"{tau_p:.4f}",
            'Krippendorff α': f"{alpha:.4f}" if not np.isnan(alpha) else 'NA'
        })

    results_df = pd.DataFrame(results)
    print("\n--- 一致性分析最终结果 ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    analyze_agreement(
        llm_filepath=LLM_SCORES_FILE,
        human_filepath=HUMAN_SCORES_FILE,
        id_col=MERGE_COLUMN_ID,
        dimensions=DIMENSIONS_TO_ANALYZE
    )
