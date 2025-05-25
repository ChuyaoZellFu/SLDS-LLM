import csv
import json
import requests
from collections import defaultdict
import random

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-ipojlltyklepynsqnilrbegjmjittmvhsosyvvfxueiuwefd"

def analyze_sentiment(prompt: str, text: str, strategy: str, idx: int) -> dict:
    """通用情感分析函数，返回预测结果和原始响应"""
    payload = {
        "model": "THUDM/GLM-Z1-9B-0414",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"评论内容：{text}"}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"}
        ).json()

        raw_response = response['choices'][0]['message']['content']
        parsed = parse_prediction(raw_response)
        
        # print(f"\n=== 第 {idx} 条评论分析结果 [{strategy}] ===")
        # print(f"原始评论：{text}...")
        # print(f"模型原始响应：\n{raw_response}")
        # print(f"解析结果：{parsed}")

        return {
            "pred": parsed,
            "raw_response": raw_response
        }
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return {
            "pred": "error",
            "raw_response": "error"
        }

def parse_prediction(text: str) -> str:
    """解析模型输出为三类情感"""
    text = text.lower()
    if any(kw in text for kw in ['强烈推荐', '谨慎好评']):
        return "positive"
    elif any(kw in text for kw in ['温和批评', '强烈批评']):
        return "negative"
    elif any(kw in text for kw in ['中立观察']):
        return "neutral"
    return "unvalid"

def load_dataset(file_path: str) -> list:
    """加载数据集并生成真实标签"""
    dataset = []
    with open(file_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                star = int(row['Star'])
                sentiment = "neutral"
                if star in (1, 2):
                    sentiment = "negative"
                elif star in (4, 5):
                    sentiment = "positive"
                dataset.append({
                    "text": row['Comment'].strip(),
                    "true": sentiment
                })
            except:
                continue
    return dataset

def evaluate(results: list) -> dict:
    """计算评估指标（含精确率、召回率、F1及准确率）"""
    strategies = set(res['strategy'] for res in results)
    classes = ['positive', 'negative', 'neutral']
    
    metrics = {}
    for strategy in strategies:
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        correct = 0  # 正确预测总数
        total = 0    # 当前策略总样本数
        num_positive = 0
        num_neutral = 0
        num_negative = 0
        pred_positive = 0
        pred_neutral = 0
        pred_negative = 0
        
        for res in results:
            if res['strategy'] != strategy:
                continue
            if res['pred'] == 'unvalid':
                continue
                
            true = res['true']
            pred = res['pred']
            total += 1  # 累计总样本
            if res['true']=='positive':
                num_positive += 1
            elif res['true']=='netural':
                num_neutral += 1
            elif res['true']=='negative':
                num_negative += 1

            if res['pred']=='positive':
                pred_positive += 1
            elif res['pred']=='netural':
                pred_neutral += 1
            elif res['pred']=='negative':
                pred_negative += 1
            
            # 全局准确率统计
            if true == pred:
                correct += 1
                
            # 类别的TP/FP/FN统计
            for c in classes:
                if pred == c:
                    if true == c:
                        tp[c] += 1
                    else:
                        fp[c] += 1
                if true == c and pred != c:
                    fn[c] += 1

        # 各类别指标计算
        precision = {}
        recall = {}
        f1 = {}
        for c in classes:
            # 精确率防零除
            denom_p = tp[c] + fp[c]
            precision[c] = tp[c] / denom_p if denom_p > 0 else 0.0
            
            # 召回率防零除
            denom_r = tp[c] + fn[c]
            recall[c] = tp[c] / denom_r if denom_r > 0 else 0.0
            
            # F1值计算
            denom_f1 = precision[c] + recall[c]
            f1[c] = 2 * (precision[c] * recall[c]) / denom_f1 if denom_f1 > 0 else 0.0

        # 准确率计算与验证
        cross_check_total = sum((tp[c] + fn[c]) for c in classes)  # 通过各类别支持数验证总数
        accuracy = correct / total if total == cross_check_total and total >0 else 0.0
        
        metrics[strategy] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': sum(precision.values())/3,
            'macro_recall': sum(recall.values())/3,
            'macro_f1': sum(f1.values())/3,
            'support': {c: tp[c]+fn[c] for c in classes},
            'accuracy': accuracy,  # 新增准确率指标
            'pred_positive': pred_positive,
            'pred_neutral': pred_neutral,
            'pred_negative': pred_negative,
            'num_positive': num_positive,
            'num_neutral': num_neutral,
            'num_negative': num_negative,
        }
    
    return metrics

if __name__ == "__main__":
    strategies = {
"chain-of-thought": 
"""请严格按以下步骤执行，并在最后用『最终结论：』明确输出结果：
1. 情感词提取（最多3个关键词语）  
2. 矛盾结构分析（仅检查1种主要矛盾）  
3. 计算综合得分（公式：Σ情感词分值×0.6 + 矛盾系数×0.4）  
4. 最终结论：根据得分输出分类：  
   >1.5→强烈推荐 | 1.0~1.5 谨慎好评 |-1.0~1.0 中立观察 | -1.5～-1.0温和批评 | <-1.5 强烈批评  """,

"role-playing": 
"""你同时担任以下两个角色，各自独立分析后投票决定最终结论：

■ 角色1：奥斯卡级影评专家  
- 专长：镜头语言、表演艺术、叙事结构  
- 输出格式：  
  【技术评分】0-10分（1分一档）  
  【结论】强烈推荐/谨慎好评/中立观察/温和批评/强烈批评   

■ 角色2：情感分析首席工程师  
- 专长：量化情感强度，检测语义冲突  
- 输出格式：  
  【情感分】-5~+5（0.5分一档）  
  【结论】强烈推荐/谨慎好评/中立观察/温和批评/强烈批评 

■ 最终决策规则：  
1. 双方结论一致时直接采纳  
2. 出现分歧时，按以下优先级选择：  
   - 差评 > 好评 > 中立（保守策略）  
   - 同等级时采纳工程师结论  

■ 必须遵守：  
- 禁止解释推理过程  
- 禁止输出评分计算细节  
- 只能从预设的5个结论中选择  
   """,
"few-shot learning": 
"""请根据以下示例的分析逻辑，直接对新的影评进行分类（仅输出最终结论）：  

【示例1】  
输入："这部电影的表演和剧本都堪称完美"  
分析：  
- 关键词：表演完美(+2.5)、剧本完美(+2.5)  
- 无负面词汇  
- 无矛盾表达  
结论：强烈推荐  

【示例2】  
输入："特效惊艳，但角色塑造完全失败"  
分析：  
- 关键词：特效惊艳(+2.0)、角色塑造失败(-2.0)  
- 存在转折词"但"（权重×1.2）  
结论：中立观察  

【示例3】  
输入："从导演到剪辑都是灾难级的"  
分析：  
- 关键词：灾难级的(-3.0)  
- 全篇无正面描述  
结论：强烈批评  

【新任务】  
输入："{{待分析的影评}}"  
请严格按示例逻辑输出结论（仅限以下选项）：  
- 强烈推荐  
- 谨慎好评  
- 中立观察  
- 温和批评  
- 强烈批评   
   """,

    }

    dataset = load_dataset("/home/slam327/SLDS-LLM/douban_movie.csv")
    print(f"已加载 {len(dataset)} 条有效评论")

    results = []
    idx = 0
    random.seed(42)
    while(True):
        random_int = random.randint(0, len(dataset)-1)
        item = dataset[random_int]
        if not item['text']:
            continue
        idx = idx + 1
        for strategy, prompt in strategies.items():
            analysis = analyze_sentiment(prompt, item['text'], strategy, idx+1)
            results.append({
                "strategy": strategy,
                "prompt": prompt,
                "comment": item['text'],
                "true": item['true'],
                "pred": analysis['pred'],
                "raw_response": analysis['raw_response']
            })
            
        print(f"已处理 {idx}/50 条")
        if idx > 49:
            break

    # 保存结果到JSON文件
    with open('/home/slam327/SLDS-LLM/API/llm_results_pd_modified_thudm.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成评估报告
    metrics = evaluate(results)
    print("\n=== 评估结果 ===")
    for strategy in strategies:
        data = metrics[strategy]
        print(f"\n策略：{strategy}")
        print(f"Num_positive:{data['num_positive']}")
        print(f"Num_neutral:{data['num_neutral']}")
        print(f"Num_negative:{data['num_negative']}")
        print(f"Pred_positive:{data['pred_positive']}")
        print(f"Pred_neutral:{data['pred_neutral']}")
        print(f"Pred_negative:{data['pred_negative']}")
        print(f"准确率：{data['accuracy']:.2%}")
        print(f"宏平均精确率：{data['macro_precision']:.2%}")
        print(f"宏平均召回率：{data['macro_recall']:.2%}")
        print(f"宏平均F1值：{data['macro_f1']:.2%}")
        
        print("\n各类别指标：")
        for c in ['positive', 'negative', 'neutral']:
            print(f"{c}:")
            print(f"  精确率：{data['precision'][c]:.2%}")
            print(f"  召回率：{data['recall'][c]:.2%}")
            print(f"  F1值：{data['f1'][c]:.2%}")
            print(f"  样本数：{data['support'][c]}")