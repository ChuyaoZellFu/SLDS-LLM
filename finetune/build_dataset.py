import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 配置参数
INPUT_FILE = "douban_movie.csv"
TRAIN_RATIO = 0.7  # 训练集比例
RANDOM_SEED = 42    # 随机种子保证可复现


def process_data():
    # 读取CSV文件，只保留 Star 和 Comment 列
    df = pd.read_csv(INPUT_FILE, usecols=["Star", "Comment"])
    # 过滤无效数据
    df = df.dropna(subset=["Star", "Comment"])  # 删除空值
    df = df[df["Star"] != 3]  # 排除中立评分
    # 转换 Star 列为数值
    df["Star"] = pd.to_numeric(df["Star"], errors="coerce")
    df = df.dropna(subset=["Star"])  # 删除无效行
    # 生成情感标签
    df["summary"] = df["Star"].apply(lambda x: "积极" if x > 3 else "消极")
    # 转换为字典列表
    data = [{"content": row["Comment"], "summary": row["summary"]} 
            for _, row in df.iterrows()]
    # 分割数据集（保持类别比例）
    train, test = train_test_split(
        data,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["summary"]  # 保持正负样本比例
    )
    return train, test


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train_data, test_data = process_data()
    save_json(train_data, "train.json")
    save_json(test_data, "test.json")
    print(f"数据集分割完成：训练集 {len(train_data)} 条，测试集 {len(test_data)} 条")