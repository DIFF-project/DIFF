import json
import os
import torch
import argparse
import pdb
import transformers

from val.StereosetRunner import Runner

from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='balance',
        # choices=['balance', 'wino'],
        help='Type of dataset to use'
    )

    parser.add_argument(
        '--percentage',
        type=str,
        default='0.02',
        help='Percentage of data to use (default: 0.02)'
    )
    
    parser.add_argument(
        '--val_type',
        type=str,
        default='pure_wino',
        choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp'],
        help='validation type: crows, stereoset, seat, wino, purewino, random'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = {
      'model_name': 'facebook/opt-350m',
      'bs': 16,
      'model_path': r'./model',
      'store_path': r"./output/stereoset",
      'data_path': r'./data/stereoset/test.json',
      'data_type': args.dataset_type,
      'val_type': args.val_type,
      'percentage': args.percentage
    }

    # model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    if config['model_path']:
        # model_path = f"../bias_IF/opt_{config['data_type']}_350m_select_{config['val_type']}/select_{config['data_type']}_small_model_{config['percentage']}_{config['val_type']}.pth"
        # model_path = f"../bias_IF/opt_{config['val_type']}_350m_select_ig/select_{config['val_type']}_small_model.pth"
        model_path = f"../bias_IF/opt_{config['val_type']}_350m_select_ig/select_{config['val_type']}_small_model_{config['percentage']}.pth"
        model.load_state_dict(torch.load(model_path))

    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]

    runner = Runner(model, tokenizer, config['model_name'], config['data_path'], config['bs'])
    results = runner()

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_stereoset_{config['val_type']}/{config['percentage']}/stereoset_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_stereoset_{config['val_type']}/{config['percentage']}/stereoset.json"
       
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/stereoset_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/stereoset.json"
    
    if config['model_path']:
      output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_stereoset_{config['val_type']}/{config['percentage']}/stereoset_ft.json"
    else:
      output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_stereoset_{config['val_type']}/{config['percentage']}/stereoset.json"  
    
    # os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
