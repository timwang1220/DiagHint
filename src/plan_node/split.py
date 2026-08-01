import numpy as np
import os
import random
import sys
from typing import List, Dict, Any

# 导入 encode_jsonl_dataset 函数
# 假设 encoder.py 在同一个目录下
from encoder import encode_jsonl_dataset

def split_jsonl_file(
    input_path: str,
    train_ratio: float = 0.8,
    output_prefix: str = "data",
    seed: int = 42
) -> (str, str):
    """
    将JSONL文件拆分为训练集和评估集。

    Args:
        input_path: 输入的JSONL文件路径。
        train_ratio: 训练集占总数据的比例。
        output_prefix: 输出文件的前缀，例如 "data" 会生成 "train-data.jsonl" 和 "eval-data.jsonl"。
        seed: 随机种子，用于可重现的拆分。

    Returns:
        一个元组，包含训练集和评估集的输出文件路径。
    """
    random.seed(seed)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 过滤掉空行
    lines = [line.strip() for line in lines if line.strip()]
    random.shuffle(lines)

    num_train = int(len(lines) * train_ratio)
    train_lines = lines[:num_train]
    eval_lines = lines[num_train:]

    train_output_path = os.path.join(os.path.dirname(input_path), f"split/train-{output_prefix}.jsonl")
    eval_output_path = os.path.join(os.path.dirname(input_path), f"split/eval-{output_prefix}.jsonl")

    with open(train_output_path, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')

    with open(eval_output_path, 'w', encoding='utf-8') as f:
        for line in eval_lines:
            f.write(line + '\n')

    print(f"已将 {input_path} 拆分为：")
    print(f"  训练集: {train_output_path} ({len(train_lines)} 行)")
    print(f"  评估集: {eval_output_path} ({len(eval_lines)} 行)")

    return train_output_path, eval_output_path


def main():
    if len(sys.argv) < 2:
        print("用法: python split.py <input_jsonl_path> [output_prefix] [train_ratio]")
        sys.exit(1)

    input_jsonl_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "data"
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    # 拆分JSONL文件
    train_file, eval_file = split_jsonl_file(input_jsonl_path, train_ratio, output_prefix)

    # 获取当前脚本所在的目录，作为 artifact 目录的基准
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 处理训练集
    train_artifact_dir = os.path.join(script_dir, f"artifacts/train-{output_prefix}-artifact")
    os.makedirs(train_artifact_dir, exist_ok=True)
    print(f"\n正在为训练集 {train_file} 调用 encode_jsonl_dataset，输出到 {train_artifact_dir}...")
    _, train_y_bucket_ids, _, _ = encode_jsonl_dataset(path=train_file, out_dir=train_artifact_dir, alias_emb_dim=64)
    print(f"训练集编码完成。")
    print("训练集每个桶的数据量:")
    for bucket_id in sorted(list(set(train_y_bucket_ids))):
        count = np.sum(train_y_bucket_ids == bucket_id)
        print(f"  桶 {bucket_id}: {count} 条数据")

    # 处理评估集
    if train_ratio == 1.0:
        return
    eval_artifact_dir = os.path.join(script_dir, f"artifacts/eval-{output_prefix}-artifact")
    os.makedirs(eval_artifact_dir, exist_ok=True)
    print(f"\n正在为评估集 {eval_file} 调用 encode_jsonl_dataset，输出到 {eval_artifact_dir}...")
    
    _, eval_y_bucket_ids, _, _ = encode_jsonl_dataset(path=eval_file, out_dir=eval_artifact_dir, alias_emb_dim=64)
    print(f"评估集编码完成。")
    print("评估集每个桶的数据量:")
    for bucket_id in sorted(list(set(eval_y_bucket_ids))):
        count = np.sum(eval_y_bucket_ids == bucket_id)
        print(f"  桶 {bucket_id}: {count} 条数据")


if __name__ == "__main__":
    main()