import json
import sys
sys.path.append("LLaMA-Factory/src")
from llamafactory.hub import get_model
from llamafactory.extras import get_infer_args

# 配置文件参数（与你的llama3.yaml一致）
CONFIG = {
    "model_name_or_path": "LLaMA-Factory/output/llama3_lora_sft_comment",
    "template": "llama3",
    "infer_backend": "huggingface",
    "trust_remote_code": True,
}

def load_model():
    """加载优化后的模型"""
    return get_model(
        get_infer_args(CONFIG),
        is_trainable=False  # 确保推理模式
    )

def predict(model, text):
    """执行单条预测"""
    prompt = f"""{text}"""
    
    response = model.generate([prompt], CONFIG)[0]
    return "积极" if "积极" in response else "消极" if "消极" in response else "unknown"

def evaluate(model):
    """执行批量评估"""
    with open("SLDS/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    results = []
    correct = 0
    i = 0
    for item in test_data:
        prediction = predict(model, item["content"])
        is_correct = (prediction == item["summary"])
        
        results.append({
            "content": item["content"],
            "true_label": item["summary"],
            "pred_label": prediction,
            "correct": is_correct
        })
        correct += int(is_correct)
        i = i + 1
        print(f"Data:{i}/{0.001*len(test_data)}")
        if i >= 0.001*len(test_data):
            break
    
    accuracy = correct / (0.001*len(test_data))
    return results, accuracy

if __name__ == "__main__":
    # 初始化模型
    model = load_model()
    
    # 执行评估
    results, accuracy = evaluate(model)
    
    # 保存结果
    with open("SLDS/result.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，准确率：{accuracy:.2%}")
    print(f"结果已保存至：SLDS/result.json")