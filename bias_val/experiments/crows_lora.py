import os
import json
import transformers
import argparse
import torch
import pdb

from val.CrowsRunner import Runner
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

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
    
    # parser.add_argument(
    #     '--dataset_type',
    #     type=str,
    #     default='balance',
    #     # choices=['balance', 'wino'],
    #     help='Type of dataset to use'
    # )

    # parser.add_argument(
    #     '--percentage',
    #     type=str,
    #     default='0.05',
    #     help='Percentage of data to use (default: 0.02)'
    # )
    
    parser.add_argument(
        '--val_type',
        type=str,
        default='pure_wino',
        # choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'pt', 'prompt_few', 'mix', 'news', 'toxic_mix'],
        help='validation type: crows, stereoset, seat, wino, pure_wino, chat_bias, random'
    )

    parser.add_argument(
        '--prompt',
        type=str2bool,
        default=False,
        help='Using retrain model or not'
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
        default="facebook/opt-1.3b",
        choices=['Qwen/Qwen2.5-1.5B-Instruct', 'facebook/opt-1.3b'],
        help='model_name'
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
      'data_path': r'./data/crows/crows_pairs_anonymized.csv' if not args.prompt else r'./data/crows/crows_pairs_prompt.csv',
      # 'data_type': args.dataset_type,
      'val_type': args.val_type,
      # 'percentage': args.percentage,
      # 'dataset_percentage': args.dataset_percentage,
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

    if args.model_name == "facebook/opt-1.3b":
      adapter_model_dir = f"../bias_sel/opt_{config['val_type']}_1.3b_select_full/peft_model_{config['val_type']}_1.3b"
    else:
      adapter_model_dir = f"../bias_sel/Qwen/Qwen_{config['val_type']}_1.5b_select_full/peft_model_{config['val_type']}_mid"

    if config['model_path']:
      model = AutoModelForCausalLM.from_pretrained(config["model_name"])
      model = PeftModel.from_pretrained(model, adapter_model_dir)

    print(adapter_model_dir)
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]
    runner = Runner(model, tokenizer, config['data_path'])
    results = runner()
       
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}/{config['percentage']}/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}/{config['percentage']}/crows.json"

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}_{config['dataset_percentage']}/{config['percentage']}/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_crows_{config['val_type']}_{config['dataset_percentage']}/{config['percentage']}/crows.json"
       
    if config['model_path']:
      output_path = f"{config['store_path']}/{name}/results_mid_{config['val_type']}_select_crows_full/crows_ft.json"
    else:
      output_path = f"{config['store_path']}/{name}/results_mid_{config['val_type']}_select_crows_full/crows.json"
       
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
