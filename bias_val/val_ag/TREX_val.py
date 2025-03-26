import transformers
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import string

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

class QueryTargetDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path) 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        prompt = """Example 1:
Particularly famous for its suggestive music video which was censored on certain TV channels, the song was a great success in France, Italy, Switzerland, Austria and <extra_id_0> where it was a top ten hit.
Missing word(s): Germany

Example 2:
Context: Sevan (Armenian: Սևան), is a town and popular resort in Armenia, located in the Gegharkunik Province on the northwestern shores of <extra_id_0>.
Missing word(s): Lake Sevan

Now fill in the blank:
Context: {query}
Missing word(s):"""

        query = self.data.iloc[idx]["query"]

        formatted_prompt = prompt.format(query=query)

        target = self.data.iloc[idx]["target"]

        inputs = self.tokenizer(
            formatted_prompt, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            target, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "query": query,
            "target": target
        }
      
def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace(" ", "")
    return text  

def evaluate_accuracy(model, dataloader, tokenizer):
    model.eval()  
    correct = 0 
    total = 0  

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        targets = batch["target"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=1,
                early_stopping=True
            )

        predicted_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for predicted, target in zip(predicted_texts, targets):
            if preprocess_text(predicted.strip()).lower() in preprocess_text(target.strip()).lower():
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    args = parse_args()

    if args.model_size == '350m':
        config = {
            'model_name': 'facebook/opt-350m',
            'model_path': f'../bias_sel/opt_{args.dataset_type}_350m_select_retrained_{args.val_type}/select_{args.dataset_type}_small_model_{args.percentage}_{args.val_type}.pth' if args.retrain else f'../bias_sel/opt_{args.dataset_type}_350m_select_ascent/ascent_{args.dataset_type}_small_model_{args.val_type}_{args.percentage}.pth',
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
    test_dataset = QueryTargetDataset(data_file='./data/ag_news/sampled_text.csv', tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    acc = evaluate_accuracy(model, test_dataloader, tokenizer)

    if args.ft:
        results_file = f"./output/TREX_results/{config['model_size']}/TREX_results_{config['data_type']}_{config['val_type']}_{config['percentage']}.txt"
    else:
        results_file = f"./output/TREX_results/{config['model_size']}/TREX_results_{config['data_type']}_ori.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"ACC: {acc}\n")