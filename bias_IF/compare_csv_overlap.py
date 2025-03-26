import pandas as pd
import os
import argparse
from tqdm import tqdm

def compare_csv_efficient(target_path, select_path):

    target_df = pd.read_csv(target_path)
    select_df = pd.read_csv(select_path)

    target_set = set(target_df.iloc[:, 0])
    select_set = set(select_df.iloc[:, 0])
    match_num = len(target_set.intersection(select_set))
    total_num = len(target_set)

    print(f"匹配数量: {match_num}")
    print(f"总数量: {total_num}")
    print(f"匹配比例: {match_num/total_num:.4f}")

    return match_num, total_num

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Script for comparing csv overlap')
    parser.add_argument("--dataset_type", type=str, default="./", help="Dataset type")
    parser.add_argument("--val_type", type=str, default="./", help="val type")
    parser.add_argument("--percentage", type=str, default="./", help="Percentage")
    parser.add_argument("--target", type=str, default="./data/350m_toxic_mix ", help="Target dataset")
    parser.add_argument("--select", type=str, default="./data/val_data/prompt_data.csv", help="Select dataset")
    args = parser.parse_args()
    target_file = os.path.join(f"{args.target}_{args.val_type}", args.percentage, f"{args.dataset_type}_dataset_select_{args.val_type}.csv")
    compare_csv_efficient(target_file, args.select)