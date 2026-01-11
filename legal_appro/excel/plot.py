import matplotlib.pyplot as plt
import numpy as np

# --- 高级字体和样式设置 ---
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16  # 全局字体更大
# plt.style.use('seaborn-whitegrid')  # 启用网格和现代样式

# --- 数据定义 ---
labels = ['Factual Accuracy', 'Legal Coherence', 'Completeness of Sentencing Factors', 'Language & Format Normality']

#judge(多被告)
# gemini_scores = [4.84, 3.53, 3.75, 4.43]
# v3_scores = [4.87, 3.37, 3.38, 4.39]
# qwen_scores = [4.77, 3.45, 3.55, 4.43]
# farui_scores = [4.45, 2.75, 3.16, 3.89]

#-------------------下面才是对的------------------------------
# # # SLJA
# gemini_scores = [4.68,4.07,4.37,4.28 ]
# v3_scores = [4.60,3.72,3.94,4.30 ]
# qwen_scores = [4.54,3.74,4.10,4.54]
# farui_scores = [4.34,3.42,3.77,3.80 ]

# #judge(单人多罪)
# gemini_scores = [4.75,3.75,4.17,4.35 ]
# v3_scores = [4.61,3.28,3.60,4.28 ]
# qwen_scores = [3.64,3.06,3.39,3.87]
# farui_scores = [4.39,3.14,3.43,3.74 ]

#cmdl()
gemini_scores = [4.59,4.20,4.33,4.20 ]
v3_scores = [4.43,3.65,3.68,4.19 ]
qwen_scores = [4.41,3.64,3.77,4.28]
farui_scores = [4.11,3.32,3.55,3.69 ]

# --- 专业配色方案 ---
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 互补色系

x = np.arange(len(labels))
width = 0.18  # 微调宽度增加间距

# --- 创建专业图表框架 ---
fig, ax = plt.subplots(figsize=(13, 8), dpi=100)
ax.set_facecolor('#f8f9fa')  # 柔和背景色

# --- 绘制增强型条形图 ---
rects1 = ax.bar(x - 1.5*width, gemini_scores, width, label='Gemini2.5', 
               color=colors[0], edgecolor='black', linewidth=0.8, zorder=3)
rects2 = ax.bar(x - 0.5*width, v3_scores, width, label='V3', 
               color=colors[1], edgecolor='black', linewidth=0.8, zorder=3)
rects3 = ax.bar(x + 0.5*width, qwen_scores, width, label='Qwen3-235B', 
               color=colors[2], edgecolor='black', linewidth=0.8, zorder=3)
rects4 = ax.bar(x + 1.5*width, farui_scores, width, label='FaRui', 
               color=colors[3], edgecolor='black', linewidth=0.8, zorder=3)

# --- 增强图表元素 ---
ax.set_ylabel('Average Score', fontsize=18, labelpad=15)
# ax.set_xlabel('Evaluation Dimensions', fontsize=18, labelpad=15,fontweight ='bold')
fig.tight_layout(pad=3.0)
# plt.suptitle('CMDL Dataset Evaluation: Average Scores of LLMs Across Dimensions', fontsize=18, fontweight='bold', y=0.98)
# X轴优化
ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=14,fontweight ='bold')
ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
# plt.subplots_adjust(bottom=0.22)  # 增加底部空间
ax.tick_params(axis='x', which='major', pad=10, labelsize=13) #这边指的是x轴的优化。

# Y轴优化
ax.set_ylim(2.0, 5.5)  # 聚焦高分区间
ax.set_yticks(np.arange(3.0, 5.6, 0.5))
ax.tick_params(axis='y', labelsize=15)

# 添加辅助网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 图例优化
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), 
          ncol=4, fontsize=12, frameon=True, shadow=True)

# --- 高级数据标签 ---
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", 
                              fc="white", ec="gray", alpha=0.8))

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# --- 添加分析注释 ---
# ax.text(0.5, 5.3, "FaRui在规范性维度表现突出", 
#         ha='center', fontsize=12, style='italic')
# ax.text(2.8, 3.2, "Qwen在情节完备性维度有提升空间", 
#         ha='center', fontsize=12, style='italic')

# 调整布局
fig.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.15)  # 为图例留空间

# # 添加版权信息
# fig.text(0.95, 0.02, '© 2025 法律AI评估中心', 
#          ha='right', va='bottom', fontsize=9, alpha=0.7)

# plt.show()

# 保存图表为pdf
fig.savefig('CMDL.pdf', dpi=300, bbox_inches='tight')

