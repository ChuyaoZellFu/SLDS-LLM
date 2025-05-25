import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# 读取JSON数据
with open('/home/slam327/SLDS-LLM/API/llm_results_pd_modified_thudm.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化性能统计字典
strategy_stats = {
    "chain-of-thought": {"TP":0, "FP":0, "FN":0, "TN":0},
    "role-playing": {"TP":0, "FP":0, "FN":0, "TN":0},
    "few-shot learning": {"TP":0, "FP":0, "FN":0, "TN":0}
}

# 统计各策略的预测结果
for item in data:
    strategy = item["strategy"]
    true_label = item["true"]
    pred_label = item["pred"]
    
    # 简化处理：将预测视为二分类（positive/非positive）
    is_true_positive = (true_label == "positive") and (pred_label == "positive")
    is_false_positive = (true_label != "positive") and (pred_label == "positive")
    is_false_negative = (true_label == "positive") and (pred_label != "positive")
    is_true_negative = (true_label != "positive") and (pred_label != "positive")
    
    if is_true_positive:
        strategy_stats[strategy]["TP"] += 1
    elif is_false_positive:
        strategy_stats[strategy]["FP"] += 1
    elif is_false_negative:
        strategy_stats[strategy]["FN"] += 1
    elif is_true_negative:
        strategy_stats[strategy]["TN"] += 1

# 计算评估指标
metrics = []
for strategy, stats in strategy_stats.items():
    TP = stats["TP"]
    FP = stats["FP"]
    FN = stats["FN"]
    TN = stats["TN"]
    
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP+FP+FN+TN)!=0 else 0
    precision = TP / (TP + FP) if (TP+FP)!=0 else 0
    recall = TP / (TP + FN) if (TP+FN)!=0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)!=0 else 0
    
    metrics.append({
        "Strategy": strategy.replace("-", " ").title(),
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2)
    })

# 创建DataFrame
df = pd.DataFrame(metrics)

# 可视化
plt.figure(figsize=(12, 6))
x = np.arange(len(df))
width = 0.2

for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    plt.bar(x + i*width, df[col], width, label=col)

plt.title("Performance Metrics Comparison by Strategy", fontsize=14)
plt.xticks(x + width*1.5, df["Strategy"])
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

# 添加数值标签
for i, row in df.iterrows():
    for j, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        plt.text(i + j*width - 0.1, row[metric]+0.02, f"{row[metric]:.2f}", 
                fontsize=9, color="black")

plt.tight_layout()
plt.show()