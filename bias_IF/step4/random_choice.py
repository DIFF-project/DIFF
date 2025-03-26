import argparse
import os

import csv
import torch
import pdb
import pandas as pd
import numpy as np
import os

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--output_path', type=str,
                           default=r"./step3/result_350m_wino", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=0.1,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()

    return args

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count

def process_csv_with_batch_indices(input_csv, output_csv, topk_indices, output_dir, percentage, batch_size=64):
    
    csv_path = os.path.join("./data", input_csv)
    df = pd.read_csv(csv_path)
    
    mask = np.ones(len(df), dtype=bool)
    
    for index in topk_indices:
        start = index * batch_size
        end = start + batch_size
        mask[start:end] = False
    
    remaining_rows = df[mask]
    
    directory = os.path.join(output_dir, percentage)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录：{directory}")
    else:
        print(f"目录已存在：{directory}")

    output_path = os.path.join(output_dir, percentage, output_csv)
    remaining_rows.to_csv(output_path, index=False)
    
    print(f"已处理完成。保留了 {len(remaining_rows)} 行，删除了 {len(df) - len(remaining_rows)} 行。")
    
def preserve_csv_with_batch_indices(input_csv, output_csv, topk_indices, output_dir, percentage, batch_size=64):
    
    csv_path = os.path.join("./data", input_csv)
    df = pd.read_csv(csv_path)
    
    mask = np.zeros(len(df), dtype=bool)
    
    for index in topk_indices:
        start = index * batch_size
        end = start + batch_size
        mask[start:end] = True
    
    remaining_rows = df[mask]
    
    directory = os.path.join(output_dir, percentage)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录：{directory}")
    else:
        print(f"目录已存在：{directory}")
    
    output_path = os.path.join(output_dir, percentage, output_csv)
    remaining_rows.to_csv(output_path, index=False)
    
    print(f"已处理完成。保留了 {len(remaining_rows)} 行，删除了 {len(df) - len(remaining_rows)} 行。")

if __name__ == "__main__":
    # pdb.set_trace()
    args = parse_args()
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    output_path = args.output_path

    score_paths = [os.path.join(
        output_path, f"opt350m_balance_influence_score.pt")]
    num_samples = []
    for score_path in score_paths:
        num_samples.append(
            len(torch.load(score_path, map_location=device)))

    total_samples = sum(num_samples)
    if args.percentage is not None:
        args.max_samples = int(args.percentage * total_samples)
        data_amount_name = f"p{args.percentage}"
    else:
        data_amount_name = f"num{args.max_samples}"

    random_indices = torch.randperm(total_samples)[:args.max_samples]
    
    # preserve_csv_with_batch_indices("balanced_dataset.csv", "balanced_dataset_random.csv", random_indices, "./data/small_balance", str(args.percentage), batch_size)
    preserve_csv_with_batch_indices("output_mid_wino.csv", "wino_dataset_random2.csv", random_indices, "./data/small_wino", str(args.percentage), batch_size)
    