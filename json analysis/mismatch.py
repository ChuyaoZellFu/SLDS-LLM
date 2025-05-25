import json
from collections import defaultdict

def analyze_mismatches(json_data):
    # 初始化存储结构
    strategy_mismatches = defaultdict(list)
    
    # 遍历每个条目
    for item in json_data:
        if item["true"] != item["pred"]:  # 检查真实值和预测值是否不同
            strategy = item["strategy"]
            strategy_mismatches[strategy].append(item["comment"]+"True:"+item["true"]+"/Pred:"+item["pred"])
    
    # 输出每种策略的不匹配评论
    for strategy, comments in strategy_mismatches.items():
        print(f"\n策略 '{strategy}' 的不匹配案例 (true≠pred):")
        for i, comment in enumerate(comments, 1):  
            print(f"{i}. {comment}")

# 示例使用
if __name__ == "__main__":
    # 加载JSON数据 (这里需要替换为实际数据加载代码)
    # 示例数据格式:
    with open('/home/slam327/SLDS-LLM/API/llm_results_pd_modified_qwen.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    analyze_mismatches(data)