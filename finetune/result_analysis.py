import json

def load_json_lines(file_path):
    """处理多行独立JSON对象（JSON Lines格式）"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"解析错误在行 {len(data)+1}: {e}")
                continue
    return data

def analyze_classification(data):
    # 初始化统计指标
    stats = {
        'total': 0,
        'correct': 0,
        'tp_pos': 0,
        'fp_pos': 0,
        'fn_pos': 0,
        'tp_neg': 0,
        'fp_neg': 0,
        'fn_neg': 0
    }
    
    for item in data:
        # 解析predict和label
        predict = item.get('predict', '')
        label = item.get('label', '').strip()
        
        # 处理label
        label_class = None
        if label in ['积极', '正面']:
            label_class = 'positive'
        elif label in ['消极', '负面']:
            label_class = 'negative'
        
        # 处理predict
        predict_class = 'neutral'
        if '正面' in predict:
            predict_class = 'positive'
        elif '负面' in predict:
            predict_class = 'negative'
        
        if not label_class:  # 无效的label跳过统计
            continue
            
        stats['total'] += 1
        
        # 统计正确数
        if (predict_class == 'positive' and label_class == 'positive') or \
           (predict_class == 'negative' and label_class == 'negative'):
            stats['correct'] += 1
        
        # 统计混淆矩阵
        # 积极类别统计
        if label_class == 'positive':
            if predict_class == 'positive':
                stats['tp_pos'] += 1
            else:
                stats['fn_pos'] += 1
        else:
            if predict_class == 'positive':
                stats['fp_pos'] += 1
                
        # 消极类别统计
        if label_class == 'negative':
            if predict_class == 'negative':
                stats['tp_neg'] += 1
            else:
                stats['fn_neg'] += 1
        else:
            if predict_class == 'negative':
                stats['fp_neg'] += 1
                
    return stats

def calculate_metrics(stats):
    metrics = {}
    
    # 准确率
    metrics['accuracy'] = stats['correct'] / stats['total'] if stats['total'] else 0
    
    # 积极类别指标
    precision_pos = stats['tp_pos'] / (stats['tp_pos'] + stats['fp_pos']) if (stats['tp_pos'] + stats['fp_pos']) > 0 else 0
    recall_pos = stats['tp_pos'] / (stats['tp_pos'] + stats['fn_pos']) if (stats['tp_pos'] + stats['fn_pos']) > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
    
    # 消极类别指标
    precision_neg = stats['tp_neg'] / (stats['tp_neg'] + stats['fp_neg']) if (stats['tp_neg'] + stats['fp_neg']) > 0 else 0
    recall_neg = stats['tp_neg'] / (stats['tp_neg'] + stats['fn_neg']) if (stats['tp_neg'] + stats['fn_neg']) > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
    
    metrics.update({
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos': f1_pos,
        'precision_neg': precision_neg,
        'recall_neg': recall_neg,
        'f1_neg': f1_neg
    })
    
    return metrics

def main(file_path):
    data = load_json_lines(file_path)
    
    if not data:
        print("未加载有效数据，请检查文件格式")
        return
    
    stats = analyze_classification(data)
    metrics = calculate_metrics(stats)
    
    # 打印结果（保持原有输出逻辑）
    print(f"总样本数: {stats['total']}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print("\n积极类别:")
    print(f"精确率: {metrics['precision_pos']:.4f}")
    print(f"召回率: {metrics['recall_pos']:.4f}")
    print(f"F1值: {metrics['f1_pos']:.4f}")
    print("\n消极类别:")
    print(f"精确率: {metrics['precision_neg']:.4f}")
    print(f"召回率: {metrics['recall_neg']:.4f}")
    print(f"F1值: {metrics['f1_neg']:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("请指定JSON文件路径")
        sys.exit(1)
    main(sys.argv[1])