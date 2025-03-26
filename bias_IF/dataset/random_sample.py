import pandas as pd
import numpy as np
import argparse
import os
def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--dataset_path', type=str,
                           default=r"./data/val_data/gen_mix_data.csv", help='The path to the output')
    argparser.add_argument('--output_prefix', type=str,
                           default=r"gen_mix", help='The path to the output')

    args = argparser.parse_args()

    return args


def sample_csv(input_file, output_prefix=None):

    df = pd.read_csv(input_file)
    
    sample_rates = [0.05]
    
    for rate in sample_rates:
        sample_size = int(len(df) * rate)
        
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        output_file = f"./data/{output_prefix}_random/{rate}/{output_prefix}_random_sample.csv"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        sampled_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(sampled_df)} rows")

args = parse_args()
dataset_path = args.dataset_path
output_prefix = args.output_prefix
sample_csv(dataset_path, output_prefix)