import json
import transformers
import torch
import pdb
import argparse

from val.CrowsRunner import Runner
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
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
        # choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'mix'],
        default='purewino',
        help='validation type: crows, stereoset, seat, wino, purewino, chatbias, random'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='Batch size'
    )

    parser.add_argument(
        '--dataset_percentage',
        type=str,
        default='full',
        help='full or few'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        choices=['Qwen/Qwen2.5-0.5B', 'facebook/opt-350m'],
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
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.model_name == 'Qwen/Qwen2.5-0.5B':
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
      'dataset_percentage': args.dataset_percentage,
      'val_type': args.val_type,
      'percentage': args.percentage
    }

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    if args.debias_type == 'retrain':
        if args.model_name == "facebook/opt-350m":
            model_path = f"../bias_sel/opt_{config['val_type']}_350m_select_retrained_{config['dataset_percentage']}/select_{config['val_type']}_small_model_{config['percentage']}_{config['dataset_percentage']}.pth"
        else:
            model_path = f"../bias_sel/Qwen/Qwen_{config['val_type']}_500m_select_retrained_{config['dataset_percentage']}/select_{config['val_type']}_small_model_{config['percentage']}_{config['dataset_percentage']}.pth"
        if config['model_path']:
            model.load_state_dict(torch.load(model_path))

    elif args.debias_type == 'npo':
        if args.model_name == "facebook/opt-350m":
            model_path = f"../bias_sel/opt_{config['val_type']}_350m_select_npo_{config['dataset_percentage']}"
        else:
            model_path = f"../bias_sel/Qwen/npo/{args.idx}_Qwen_{config['val_type']}_500m_select_npo_{config['dataset_percentage']}"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    
    print(model_path)
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    # eos_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token_id = [str(eos_token_id)]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    runner = Runner(model, tokenizer, config['data_path'])
    results = runner()

    if config['model_path']:
      output_path = f"{config['store_path']}/{name}/results_small_{config['val_type']}_select_{config['val_type']}_{config['dataset_percentage']}/{config['percentage']}/crows_ft.json"
    else:
      output_path = f"{config['store_path']}/{name}/results_small_{config['val_type']}_select_{config['val_type']}_{config['dataset_percentage']}/{config['percentage']}/crows.json"
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_select_{config['val_type']}/{config['percentage']}/crows_full.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_select_{config['val_type']}/{config['percentage']}/crows.json"
      
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/crows.json"
       
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
