import os
import re
import json
import transformers
import argparse
import torch
import pdb
import csv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from openai import OpenAI
import openai

from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai.verify_ssl = False
client = OpenAI(api_key="Rrija43BjeyobHtz1myrwChxlVQH2BWs", base_url="https://api.deepinfra.com/v1/openai")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='mix',
        help='Type of dataset to use'
    )

    parser.add_argument(
        '--percentage',
        type=str,
        default='0.05',
        help='Percentage of data to use (default: 0.02)'
    )
    
    parser.add_argument(
        '--val_type',
        type=str,
        default='few',
        help='validation type: few or full'
    )

    parser.add_argument(
        '--model_size',
        type=str,
        default='350m',
        help='model size: 350m or 1.3b'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size'
    )

    parser.add_argument(
        '--ft',
        type=str2bool,
        default=True,
        help='Using finetune model or not'
    )
    
    args = parser.parse_args()
    return args

class AGNewsDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=512):
        self.data = pd.read_csv(data_file, header=None)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def generate_title(self, model, idx, max_length=300):
        text = self.get_text(idx)
        inputs = self.tokenizer("Generate a one line title for: " + text, 
                              return_tensors="pt",
                              truncation=True,
                              max_length=self.max_len)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_title

    def get_text(self, idx):
        return self.data.iloc[idx].item()
    def __getitem__(self, idx):
        # text = self.data.iloc[idx]["title"] + " " + self.data.iloc[idx]["content"]
        text = self.data.iloc[idx].item()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def calculate_rouge_score_sent(target_title, generated_title):

    def get_lcs_length(str1, str2):
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    target_words = target_title.strip().split()
    generated_words = generated_title.strip().split()
    
    if len(target_words) == 0 or len(generated_words) == 0:
        return 0
    
    lcs_length = get_lcs_length(target_words, generated_words)
    
    precision = lcs_length / len(generated_words)
    recall = lcs_length / len(target_words)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return f1_score
    
def calculate_rouge_score_all(dataset, model, csv_file="./data/ag_news/target_titles.csv"):
    system_message = """Generate ONLY a one-line title for the given article. 
    Do not include any thinking process, explanations, or additional text.
    Output format should be exactly one line containing only the title."""
    message = [{"role": "system", "content": system_message}]
    total_score = 0

    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            target_titles = [row[0] for row in reader]
    else:
        target_titles = []

    for idx in tqdm(range(len(dataset))):
        if idx < len(target_titles):
            target_title = target_titles[idx]
        else:
            if len(message) <= 1:
                message.append({"role": "user", "content": dataset.get_text(idx)})
            else:
                message[-1] = {"role": "user", "content": dataset.get_text(idx)}

            print(f"query: {dataset.get_text(idx)}")
            target_title = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=message,
                stream=False,
                temperature=0.7
            )
            content = target_title.choices[0].message.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            target_title = re.sub(r'\s+', ' ', content).strip()
            
            print(f"target_title: {target_title}")

            target_titles.append(target_title)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([target_title])
                file.flush()

        # finetune_title = dataset.generate_title(model, idx)
        finetune_title = 'abc'
        print(f"finetune_title: {finetune_title}")

        # total_score += calculate_rouge_score_sent(target_title, finetune_title)

    # total_score /= len(dataset)

    return total_score

# def calculate_rouge_score_all(dataset, model):

#     system_message = "You should generate a one line title for given article. You should and only allowed to generate the one line title without any thinking process."
#     message = [{"role": "system", "content": system_message}]
#     total_score = 0

#     for idx in tqdm(len(dataset)):
#         # generate title using finetune model
#         if len(message) > 1:
#             message.append({"role": "user", "content": dataset.get_text(idx)})
#         else:
#             message[-1] = {"role": "user", "content": dataset.get_text(idx)}

#         finetune_title = dataset.generate_title(model, idx)

#         # generate title using powerful model
#         target_title = client.chat.completions.create(
#             model="deepseek-ai/DeepSeek-R1",
#             messages=message,
#             stream=False,
#             temperature=0.7
#         )
#         target_title = target_title.choices[0].message.content

#         total_score += calculate_rouge_score_sent(target_title, fintune_title)

#     total_score /= len(dataset)

#     return total_score

if __name__ == "__main__":
    args = parse_args()

    if args.model_size == '350m':
        config = {
            'model_name': 'facebook/opt-350m',
            'model_path': f'../bias_sel/opt_{args.dataset_type}_350m_select_retrained_{args.val_type}/select_{args.val_type}_small_model_{args.percentage}.pth',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage
        }
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        if args.ft:
            model.load_state_dict(torch.load(config['model_path']))

    else:
        config = {
            'model_name': 'facebook/opt-1.3b',
            'model_path': f'../bias_sel/opt_{args.dataset_type}_1.3b_select_retrained_{args.val_type}/select_{args.percentage}_peft_model_{args.dataset_type}_{args.val_type}',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage
        }

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        model = PeftModel(model, lora_config)
        if args.ft:
            model = PeftModel.from_pretrained(model, config["model_path"])
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # test_dataset = AGNewsDataset(data_file='./data/test.csv', tokenizer=tokenizer)
    test_dataset = AGNewsDataset(data_file='./data/ag_news/toxic_mix_data.csv', tokenizer=tokenizer)

    rouge_score = calculate_rouge_score_all(test_dataset, model)

    print(f"rouge_score is: {rouge_score}")