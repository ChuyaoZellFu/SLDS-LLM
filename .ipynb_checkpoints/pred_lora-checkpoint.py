import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

BASE_MODEL = "/mnt/workspace/LLaMA-Factory/Meta-Llama-3-8B-Instruct"  # 基础模型名称或路径
LORA_PATH = "/mnt/workspace/LLaMA-Factory/saves/llama3-8b/lora/sft_comment"        # LoRA 适配器路径
DATA_PATH = "/mnt/workspace/SLDS/test.json"             # 输入数据路径
RESULT_PATH = "/mnt/workspace/SLDS/result.json"               # 结果保存路径

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",          # 自动分配多GPU
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    adapter_name="sentiment_lora"
)

model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.eval()

def load_data():
    """加载并验证数据格式（与原函数相同）"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    required_keys = {"content", "summary"}
    for item in data:
        if not required_keys.issubset(item.keys()):
            raise ValueError("Invalid data format")
    return data

def predict(text):
    """生成预测结果（优化prompt模板）"""
    prompt = f"""{text}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,          # 限制输出长度
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "积极" if "积极" in response else "消极" if "消极" in response else response
def evaluate(data):
    """执行评估（与原函数相同）"""
    results = []
    correct = 0
    
    i = 0
    for item in data:
        prediction = predict(item["content"])
        is_correct = (prediction == item["summary"])
        result = {
            "content": item["content"],
            "true_label": item["summary"],
            "pred_label": prediction,
            "correct": is_correct
        }
        results.append(result)
        correct += int(is_correct)
        i = i + 1
        print(f"Data {i}/{0.001*len(data)}")
        if i >= 0.001*len(data):
            break
    
    accuracy = correct / (0.001*len(data))
    return results, accuracy

if __name__ == "__main__":
    data = load_data()
    results, accuracy = evaluate(data)
    
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，准确率：{accuracy:.2%}")
    print(f"详细结果已保存至：{RESULT_PATH}")