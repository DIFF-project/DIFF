import os
import json
import transformers
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from val.CrowsRunner import Runner
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='balance',
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
        default='pure_wino',
        choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'pt', 'prompt_few'],
        help='validation type: crows, stereoset, seat, wino, pure_wino, chat_bias, random'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size'
    )
    
    args = parser.parse_args()
    return args

class AGNewsDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=512):
        self.data = pd.read_csv(data_file, header=None, names=["label", "title", "content"])
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]["title"] + " " + self.data.iloc[idx]["content"]
        label = self.data.iloc[idx]["label"]
        
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
            'label': torch.tensor(label, dtype=torch.long)
        }
        
def test(model, dataloader):
    model.eval() 
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad(): 
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions.double() / total_predictions
    return accuracy

if __name__ == "__main__":
    args = parse_args()
    config = {
      'model_name': 'facebook/opt-1.3b',
      'model_path': r'./model',
      'store_path': r"./output/crows",
      'data_path': r'./data/crows/crows_pairs_anonymized.csv',
      'data_type': args.dataset_type,
      'val_type': args.val_type,
      'percentage': args.percentage,
    }
    
    lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.CAUSAL_LM
    )

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = PeftModel(model, lora_config)

    adapter_model_dir = adapter_model_dir = f"../bias_IF/opt_{config['val_type']}_1.3b_select_ig/select_{config['percentage']}_peft_model_{config['val_type']}"

    if config['model_path']:
      model = AutoModelForCausalLM.from_pretrained(config["model_name"])
      model = PeftModel.from_pretrained(model, adapter_model_dir)

    print(adapter_model_dir)
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]
    
    test_dataset = AGNewsDataset(data_file='./data/test.csv', tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    accuracy = test(model, test_dataloader)
    print(f"Test Accuracy: {accuracy:.4f}")

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}/{config['percentage']}/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}/{config['percentage']}/crows.json"
       
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
