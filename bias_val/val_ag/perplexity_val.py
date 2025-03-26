import os
import json
import transformers
import argparse
import torch
import pdb
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--ft',
        type=str2bool,
        default=True,
        help='Using finetune model or not'
    )
    
    parser.add_argument(
        '--retrain',
        type=str2bool,
        default=False,
        help='Using retrain model or not'
    )

    args = parser.parse_args()
    return args

# TODO 换成最原始的公式
# def calculate_batch_perplexity(model, batch, loss_func, device='cuda'):
#     with torch.no_grad():
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         # pdb.set_trace()
        
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         # loss = outputs.loss
#         loss = selfloss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         print(f"current loss: {loss}")
#         perplexity = torch.exp(loss)
#         return perplexity.item()
def calculate_batch_perplexity(model, batch, loss_func, device='cuda'):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        labels = input_ids[:, 1:]
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()
        
        loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # print(f"current loss: {loss}")
        perplexity = torch.exp(loss)
        return perplexity.item()


# def calculate_batch_perplexity(model, batch, device='cuda'):
#     with torch.no_grad():
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
        
#         labels = input_ids.clone()
#         labels = torch.roll(labels, shifts=-1, dims=1)
#         labels[:, -1] = -100
        
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         loss = outputs.loss
        
#         perplexity = torch.exp(loss)
#         pdb.set_trace()
#         return perplexity.item()

def evaluate_perplexity(model, dataloader, device='cuda'):
    model.eval()
    model = model.to(device)
    
    total_perplexity = 0
    batch_count = 0
    perplexities = []
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=2)
    
    with tqdm(total=len(dataloader), desc="Calculating Perplexity") as pbar:
        for batch in dataloader:
            batch_perplexity = calculate_batch_perplexity(model, batch, loss_func, device)
            perplexities.append(batch_perplexity)
            
            total_perplexity += batch_perplexity
            batch_count += 1
            pbar.update(1)
    
    mean_perplexity = total_perplexity / batch_count
    std_perplexity = np.std(perplexities)
    
    return {
        'mean_perplexity': mean_perplexity,
        'std_perplexity': std_perplexity,
        'all_perplexities': perplexities
    }

class AGNewsDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=512):
        self.data = pd.read_csv(data_file, header=None)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx].item()
        
        # print(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:] 
        labels[-1] = 2
        
        labels[attention_mask == 0] = 2
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# class AGNewsDataset(Dataset):
#     def __init__(self, data_file, tokenizer, max_len=512):
#         self.data = pd.read_csv(data_file, header=None, names=["label", "title", "content"])
#         self.tokenizer = tokenizer
#         self.max_len = max_len
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         text = self.data.iloc[idx]["title"] + " " + self.data.iloc[idx]["content"]
        
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
        
#         input_ids = encoding['input_ids'].squeeze()
#         labels = input_ids.clone()
        
#         labels[encoding['attention_mask'].squeeze() == 0] = -100
        
#         labels = torch.roll(labels, shifts=-1, dims=0)
#         labels[-1] = -100
        
#         return {
#             'input_ids': input_ids,
#             'attention_mask': encoding['attention_mask'].squeeze(),
#             'labels': labels
#         }
        
#     # def __getitem__(self, idx):
#     #     text = self.data.iloc[idx]["title"] + " " + self.data.iloc[idx]["content"]
        
#     #     encoding = self.tokenizer.encode_plus(
#     #         text,
#     #         add_special_tokens=True,
#     #         max_length=self.max_len,
#     #         padding='max_length',
#     #         truncation=True,
#     #         return_attention_mask=True,
#     #         return_tensors='pt'
#     #     )
        
#     #     return {
#     #         'input_ids': encoding['input_ids'].flatten(),
#     #         'attention_mask': encoding['attention_mask'].flatten(),
#     #     }

if __name__ == "__main__":
    args = parse_args()

    if args.model_size == '350m':
        config = {
            'model_name': 'facebook/opt-350m',
            'model_path': f'../bias_sel/opt_{args.dataset_type}_350m_select_retrained_{args.val_type}/select_{args.dataset_type}_small_model_{args.percentage}_{args.val_type}.pth' if args.retrain else f'../bias_sel/opt_{args.dataset_type}_350m_select_ascent/ascent_{args.dataset_type}_small_model_{args.val_type}_{args.percentage}.pth',
            # 'model_path': f'../bias_sel/opt_{args.dataset_type}_350m_select_full/select_{args.val_type}_small_model.pth',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': args.model_size
        }

        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        if args.ft:
            model.load_state_dict(torch.load(config['model_path']))

    else:
        config = {
            'model_name': 'facebook/opt-1.3b',
            # 'model_path': f'../bias_sel/opt_{args.dataset_type}_1.3b_select_retrained_{args.val_type}/select_{args.percentage}_peft_model_{args.dataset_type}_{args.val_type}',
            'model_path': f'../bias_sel/opt_{args.dataset_type}_1.3b_select_retrained_{args.val_type}/select_{args.percentage}_peft_model_{args.dataset_type}_{args.val_type}' if args.retrain else f'../bias_sel/opt_{args.dataset_type}_1.3b_select_ascent/ascent_{args.dataset_type}_mid_model_{args.val_type}_{args.percentage}',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': args.model_size
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
        model = model.merge_and_unload()
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = AGNewsDataset(data_file='./data/ag_news/sampled_text.csv', tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    perplexity_results = evaluate_perplexity(model, test_dataloader, device)
    
    print(f"Mean Perplexity: {perplexity_results['mean_perplexity']:.2f}")
    print(f"Std Perplexity: {perplexity_results['std_perplexity']:.2f}")
    
    if args.ft:
        results_file = f"./output/perplexity_results/{config['model_size']}/perplexity_results_{config['data_type']}_{config['val_type']}_{config['percentage']}.txt"
    else:
        results_file = f"./output/perplexity_results/{config['model_size']}/perplexity_results_{config['data_type']}_ori.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Mean Perplexity: {perplexity_results['mean_perplexity']:.2f}\n")
        f.write(f"Std Perplexity: {perplexity_results['std_perplexity']:.2f}\n")