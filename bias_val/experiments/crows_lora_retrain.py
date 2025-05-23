import os
import json
import transformers
import argparse
import torch
import pdb

from val.CrowsRunner import Runner
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
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
        default='trex',
        # choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'pt', 'prompt_few', 'mix'],
        help='validation type: crows, stereoset, seat, wino, pure_wino, chat_bias, random'
    )

    parser.add_argument(
        '--dataset_percentage',
        type=str,
        default='full',
        help='full or few'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        choices=['Qwen/Qwen2.5-1.5B-Instruct', 'facebook/opt-1.3b'],
        help='model_name'
    )

    parser.add_argument(
        '--debias_type',
        type=str,
        default="retrain",
        choices=['retrain', 'npo'],
        help='model_name'
    )

    parser.add_argument(
        '--idx',
        type=int,
        default=14,
    )

    parser.add_argument(
        '--step',
        type=int,
        default=7820,
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.model_name == 'Qwen/Qwen2.5-1.5B-Instruct':
        name = 'Qwen'
        num = 500
    else:
        name = 'opt'
        num = 350

    config = {
      'model_name': args.model_name,
      'model_path': r'./model',
      'store_path': r"./output/crows",
      'data_path': r'./data/crows/crows_pairs_anonymized.csv',
      'val_type': args.val_type,
      'percentage': args.percentage,
      'dataset_percentage': args.dataset_percentage
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

    # adapter_model_dir = adapter_model_dir = f"../bias_IF/opt_{config['val_type']}_1.3b_select_ig/peft_model_{config['val_type']}_1.3b"
    
    if args.debias_type == 'retrain':
      if args.model_name == "facebook/opt-1.3b":
        adapter_model_dir = f"../bias_sel/opt_{config['val_type']}_1.3b_select_retrained_{config['dataset_percentage']}/select_{config['percentage']}_peft_model_{config['val_type']}_{config['dataset_percentage']}"
      else:
        adapter_model_dir = f"../bias_sel/Qwen/Qwen_{config['val_type']}_1.5b_select_retrained_{config['dataset_percentage']}/select_{config['percentage']}_peft_model_{config['val_type']}_{config['dataset_percentage']}"
      
      if config['model_path']:
        print(adapter_model_dir)
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        model = PeftModel.from_pretrained(model, adapter_model_dir)
    
    elif args.debias_type == 'npo':
        if args.model_name == "facebook/opt-350m":
            model_path = f"../bias_sel/opt_{config['val_type']}_1.3b_select_npo_{config['dataset_percentage']}"
        else:
            # model_path = f"../bias_sel/Qwen/npo/{args.idx}_Qwen_{config['val_type']}_1.5b_select_npo_{config['dataset_percentage']}"
            model_path = f"../bias_sel/Qwen/npo/{args.idx}_Qwen_{config['val_type']}_1.5b_select_npo_{config['dataset_percentage']}/{args.idx}-checkpoint/checkpoint-{args.step}"
            # model_path = "../bias_sel/Qwen/npo/14_Qwen_trex_1.5b_select_npo_few/14-checkpoint/checkpoint-7820"
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    # eos_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token_id = [str(eos_token_id)]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    runner = Runner(model, tokenizer, config['data_path'])
    results = runner()

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows.json"
       
    if config['model_path']:
      output_path = f"{config['store_path']}/{name}/results_mid_select_crows_{config['val_type']}_{config['dataset_percentage']}_retrained/{config['percentage']}/crows_ft.json"
    else:
      output_path = f"{config['store_path']}/{name}/results_mid_select_crows_{config['val_type']}_{config['dataset_percentage']}_retrained/{config['percentage']}/crows.json"
       
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['val_type']}_select_crows_ig/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['val_type']}_select_crows_ig/crows.json"
       
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
