import argparse
import os

import numpy as np
import torch
import pdb

# 这里只写了一个ckpt

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--model_name', type=str, default="Qwen",
                       help='Qwen or opt')
argparser.add_argument('--val_type', type=str, default="crows",
                       help='crows seat stereoset')
argparser.add_argument('--train_type', type=str, default="wino",
                       help='wino_or_balance')
argparser.add_argument('--model_size', type=str, default="350m",
                       help='1.3b or 350m')
argparser.add_argument('--dataset_percentage', type=str, default='full',
                       help='Whether using full dataset or not')

args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):

    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

if args.model_name == 'opt':
    validation_path = f"./step2/result_{args.model_size}_{args.train_type}_val_{args.dataset_percentage}/dim8192"
else: 
    validation_path = f"./step2/Qwen/result_{args.model_size}_{args.train_type}_val_{args.dataset_percentage}/dim8192"

if os.path.isdir(validation_path):
    validation_path = os.path.join(validation_path, "all_orig.pt")
validation_info = torch.load(validation_path)

if not torch.is_tensor(validation_info):
    validation_info = torch.tensor(validation_info)
validation_info = validation_info.to(device).float()

if args.model_name == 'opt':
    gradient_path = f"./step2/result_{args.model_size}_{args.train_type}_{args.dataset_percentage}/dim8192"
else:
    gradient_path = f"./step2/Qwen/result_{args.model_size}_{args.train_type}_{args.dataset_percentage}/dim8192"

if os.path.isdir(gradient_path):
    gradient_path = os.path.join(gradient_path, "all_orig.pt")
training_info = torch.load(gradient_path)

if not torch.is_tensor(training_info):
    training_info = torch.tensor(training_info)
training_info = training_info.to(device).float()

influence_score = calculate_influence_score(training_info=training_info, validation_info=validation_info)

influence_score = influence_score.reshape(influence_score.shape[0], 1, -1).mean(-1).max(-1)[0]

if args.model_name == 'opt':
    output_dir = f"./step3/result_{args.model_size}_{args.train_type}_{args.dataset_percentage}"
else:
    output_dir = f"./step3/Qwen/result_{args.model_size}_{args.train_type}_{args.dataset_percentage}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, f"opt350m_balance_influence_score.pt")
torch.save(influence_score, output_file)
print("Saved influence score to {}".format(output_file))
