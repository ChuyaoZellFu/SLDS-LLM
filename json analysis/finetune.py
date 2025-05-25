import jsonlines
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 初始化统计变量
TP = FP = TN = FN = 0

with jsonlines.open('/home/slam327/SLDS-LLM/generated_predictions_base.jsonl') as reader:
    for obj in reader:
        # 标签转换
        y_true = 1 if obj['label'].strip() == '积极' else 0
        y_pred = 0 if 'Negative' in obj['predict'].strip() else 1
        
        # 更新四象限统计
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1
        else:
            FN += 1

# 计算评估指标
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 可视化输出
print(f'''
分类性能报告：
├── 准确率（Accuracy）: {accuracy:.2%}
├── 精确率（Precision）: {precision:.2%} 
├── 召回率（Recall）: {recall:.2%}
└── F1分数: {f1:.2%}

混淆矩阵：
        预测积极  预测消极
真实积极   {TP:^5}    {FN:^5}
真实消极   {FP:^5}    {TN:^5}
''')

# 创建画布和子图布局
plt.figure(figsize=(12, 5))
# plt.rcParams['font.family'] = 'SimHei'  # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False

# 子图1：混淆矩阵热力图
plt.subplot(1, 2, 1)
conf_matrix = np.array([[TP, FN], [FP, TN]])
sns.heatmap(conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='GnBu',
            annot_kws={'size':14},
            cbar=False)
plt.title('conf_matrix', fontsize=14, pad=20)
plt.xlabel('pred', fontsize=12)
plt.ylabel('label', fontsize=12)
plt.xticks([0.5,1.5], ['active', 'negative'])
plt.yticks([0.5,1.5], ['active', 'negative'], rotation=0)

# 子图2：评估指标柱状图
plt.subplot(1, 2, 2)
metrics = ['Accuarcy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = plt.bar(metrics, 
               values, 
               color=colors,
               alpha=0.8,
               edgecolor='black')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height-0.05,
             f'{height:.2%}',
             ha='center', 
             va='bottom',
             color='white',
             fontsize=12)

plt.ylim(0, 1.1)
plt.title('Comparision', fontsize=14, pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()