import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义数据
error_types = [
    'Incorrect Application of Articles',
    'Incorrect Charge Determination',
    'Logical Contradiction',
    'Confusion of Legal Concepts',
    'Omission of Charges',
    'Hallucination of Legal Articles'
]

v3_freq = [32, 26, 7, 6, 6, 4]
farui_freq = [34, 26, 8, 9, 0, 1]

# 2. 准备Farui的数据 (过滤掉频率为0的项)
farui_labels = [label for i, label in enumerate(error_types) if farui_freq[i] > 0]
farui_data = [freq for freq in farui_freq if freq > 0]

# 3. 创建图形 - 调整尺寸比例
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

# 4. 设置统一的颜色方案
colors = plt.cm.tab10.colors[:len(error_types)]

# 5. 绘制 V3 的饼图 (左侧)
wedges1, texts1, autotexts1 = ax1.pie(
    v3_freq, 
    labels=error_types, 
    autopct=lambda p: f'{p:.1f}%', 
    startangle=90,
    colors=colors,
    wedgeprops={'edgecolor': 'w', 'linewidth': 1.5},
    textprops={'fontsize': 14, 'color': 'black', 'fontname': 'Times New Roman','weight': 'bold'},
    pctdistance=0.85,
    labeldistance=0.95  # 标签更靠近扇区
)
plt.setp(texts1, fontsize=13, fontname='Times New Roman', wrap=True)
plt.setp(autotexts1, fontsize=12, weight='bold', color='white', fontname='Times New Roman')
ax1.set_title('V3 Error Distribution', fontsize=18, pad=15, fontname='Times New Roman',fontweight ='bold')
ax1.axis('equal')

# 6. 绘制 Farui 的饼图 (右侧)
wedges2, texts2, autotexts2 = ax2.pie(
    farui_data, 
    labels=farui_labels, 
    autopct=lambda p: f'{p:.1f}%',
    startangle=90,
    colors=[c for i, c in enumerate(colors) if farui_freq[i] > 0],
    wedgeprops={'edgecolor': 'w', 'linewidth': 1.5},
    textprops={'fontsize': 14, 'color': 'black', 'fontname': 'Times New Roman','weight': 'bold'},
    pctdistance=0.85,
    labeldistance=0.95
)
plt.setp(texts2, fontsize=13, fontname='Times New Roman', wrap=True)
plt.setp(autotexts2, fontsize=12, weight='bold', color='white', fontname='Times New Roman')
ax2.set_title('Farui Error Distribution', fontsize=18, pad=15, fontname='Times New Roman',fontweight ='bold')
ax2.axis('equal')

# 7. 添加主标题
# fig.suptitle('Comparison of Error Distributions: V3 vs. Farui', 
#              fontsize=22, y=0.98, weight='bold', fontname='Times New Roman')

# 8. 添加图例说明
# fig.text(0.5, 0.02, 
#          'Note: Farui has zero "Omission of Charges" errors.',
#          ha='center', fontsize=13, style='italic', fontname='Times New Roman')

# 9. 优化布局
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.subplots_adjust(wspace=0.3)

plt.show()
#保存为pdf
fig.savefig('error_distribution_comparison.pdf', bbox_inches='tight', dpi=300)