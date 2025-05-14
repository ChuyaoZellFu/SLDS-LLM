import json
import argparse

def process_data(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每条数据
    processed_data = []
    for item in data:
        # 构建新的 content 格式
        new_content = (
            "你是一个分类模型，请判断以下文本的情感倾向：\n"
            f"输入：{item['content'].strip()}\n"
            "选项：正面或负面\n"
            "答案："
        )
        
        # 保持原有结构，只修改 content
        processed_item = {
            "content": new_content,
            "summary": item["summary"]
        }
        processed_data.append(processed_item)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理情感分析数据集')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    try:
        process_data(args.input, args.output)
        print(f"数据处理完成，已保存至 {args.output}")
    except Exception as e:
        print(f"处理失败：{str(e)}")