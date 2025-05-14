import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 配置参数
MODEL_PATH = "LLaMA-Factory/output/llama3_lora_sft_comment"  # 本地模型路径
DATA_PATH = "SLDS/test.json"                 # 输入数据路径
RESULT_PATH = "SLDS/result.json"             # 结果保存路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# 关键参数1：量化配置
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,          # 启用8-bit量化
    llm_int8_threshold=6.0,     # 异常值检测阈值（默认6.0）
    llm_int8_skip_modules=[],  # 指定不量化的模块
    llm_int8_enable_fp32_cpu_offload=False  # 是否启用CPU卸载
)

# 关键参数2：设备映射
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"           # 自动分配设备
)

# 关键参数3：修复pad_token（如果未定义）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_data():
    """加载并验证数据格式"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 验证数据格式
    required_keys = {"content", "summary"}
    for item in data:
        if not required_keys.issubset(item.keys()):
            raise ValueError("Invalid data format")
    return data


def predict(text):
    """生成预测结果"""
    prompt = f"""{text}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.01,         # 降低随机性
            do_sample=False           # 使用贪婪解码
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "积极" if "积极" in response else "消极" if "消极" in response else "unknown"


def evaluate(data):
    """执行评估"""
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
        print(f"Data:{i}/{0.001*len(data)}")
        if i >= 0.001*len(data):
            break

    accuracy = correct / (0.001*len(data))
    return results, accuracy


if __name__ == "__main__":
    # 执行流程
    data = load_data()
    results, accuracy = evaluate(data)
    # 保存结果
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, ensure_ascii=False, indent=2)
    print(f"评估完成，准确率：{accuracy:.2%}")
    print(f"详细结果已保存至：{RESULT_PATH}")