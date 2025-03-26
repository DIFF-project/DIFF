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
    # argparser.add_argument('--grad_path', type=str,
    #                        default=r"./step3/result_1.3b_balance_stereoset", help='The path to the grad')
    argparser.add_argument('--model_size', type=str,
                           default=r"350m", help='350m or 1.3b')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=0.02,
                           help='The percentage of the data to be selected')
    argparser.add_argument('--test_set', type=str, default='stereoset',
                           help='validation_test_set')
    # argparser.add_argument('--select_dataset', type=str, default="balanced_dataset.csv",
    #                        help='where the data selected from')
    argparser.add_argument('--model_name', type=str, default="Qwen",
                       help='Qwen or opt')
    argparser.add_argument('--val_type', type=str, default="balance",
                           help='where the data selected from')
    argparser.add_argument('--bs', type=int, default=4,
                           help='batch size')
    argparser.add_argument('--dataset_percentage', type=str, default='full',
                       help='Whether using full dataset or not') 
    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


def process_csv_with_batch_indices(input_csv, output_csv, topk_indices, output_dir, percentage, batch_size=64):
    
    csv_path = os.path.join("./data/val_data", input_csv)
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
    
    csv_path = os.path.join("./data/val_data", input_csv)
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

    batch_size = args.bs

    if args.model_name == 'opt':
        output_path = f"./step3/result_{args.model_size}_{args.val_type}_{args.dataset_percentage}"
    else:
        output_path = f"./step3/Qwen/result_{args.model_size}_{args.val_type}_{args.dataset_percentage}"

    score_paths = [os.path.join(
        output_path, f"opt350m_balance_influence_score.pt")]
    num_samples = []
    for score_path in score_paths:
        num_samples.append(
            len(torch.load(score_path, map_location=device)))
    cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)
    
    total_samples = sum(num_samples)
    if args.percentage is not None:
        args.max_samples = int(args.percentage * total_samples)
        data_amount_name = f"p{args.percentage}"
    else:
        data_amount_name = f"num{args.max_samples}"

    all_scores = []
    for score_path in score_paths:
        score = torch.load(score_path, map_location=device)
        all_scores.append(score)
    all_scores = torch.cat(all_scores, dim=0)

    file_specific_index = torch.cat(
        [torch.arange(line_num) for line_num in num_samples]).to(device)
    
    data_from_list = []
    for i, line_num in enumerate(num_samples):
        tensor = torch.ones(line_num, dtype=torch.long) * i
        data_from_list.append(tensor)
    data_from = torch.cat(data_from_list).to(device)

    # data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
    #                         * i for i, line_num in enumerate(num_samples)]).to(device)
    sorted_scores, sorted_index = torch.sort(
        all_scores, dim=0, descending=True)
    sorted_score_file = os.path.join(output_path, f"sorted.csv")

    data_from = data_from[sorted_index]
    sorted_index = file_specific_index[sorted_index]
    # pdb.set_trace()

    if not os.path.exists(sorted_score_file):
        with open(sorted_score_file, 'w', encoding='utf-8') as file:
            file.write("index, score\n")
            for score, index, name in zip(sorted_scores, sorted_index, data_from):
                file.write(
                    f"{index.item()}, {round(score.item(), 6)}\n")

    topk_scores, topk_indices = torch.topk(
        all_scores.float(), args.max_samples, dim=0, largest=True)
    
    if args.val_type in ["balance", "prompt", "mix", "toxic_mix", 'gen_mix', 'trex']:
        select_dataset = f"{args.val_type}_data.csv"
    else:
        select_dataset = "output_mid_wino.csv"

    process_csv_with_batch_indices(select_dataset, f"{args.val_type}_dataset_not_select_{args.dataset_percentage}.csv", topk_indices, f"./data/{args.model_name}/{args.model_size}_{args.val_type}_{args.dataset_percentage}", str(args.percentage), batch_size)
    preserve_csv_with_batch_indices(select_dataset, f"{args.val_type}_dataset_select_{args.dataset_percentage}.csv", topk_indices, f"./data/{args.model_name}/{args.model_size}_{args.val_type}_{args.dataset_percentage}", str(args.percentage), batch_size)
    # process_csv_with_batch_indices(select_dataset, f"balance_dataset_not_select_{args.test_set}.csv", topk_indices, f"./data/{args.model_size}_{args.val_type}_{args.test_set}/bs{args.bs}", str(args.percentage), batch_size)
    # preserve_csv_with_batch_indices(select_dataset, f"balance_dataset_select_{args.test_set}.csv", topk_indices, f"./data/{args.model_size}_{args.val_type}_{args.test_set}/bs{args.bs}", str(args.percentage), batch_size)