# import os
# import json
# import transformers
# import torch
# import argparse

# from val.CrowsRunner import Runner
# from transformers import AutoModelForCausalLM

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Configuration for validation and dataset selection'
#     )
    
#     parser.add_argument(
#         '--dataset_type',
#         type=str,
#         default='balance',
#         choices=['balance', 'wino'],
#         help='Type of dataset to use'
#     )

#     parser.add_argument(
#         '--percentage',
#         type=str,
#         default='0.02',
#         help='Percentage of data to use (default: 0.02)'
#     )
    
#     parser.add_argument(
#         '--bs',
#         type=int,
#         default=16,
#         help='Batch size'
#     )
    
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args = parse_args()
#     config = {
#       'model_name': 'facebook/opt-350m',
#       'model_path': r'./model',
#       'store_path': r"./output/crows",
#       'data_path': r'./data/crows/crows_pairs_anonymized.csv',
#       'data_type': args.dataset_type,
#       'val_type': 'crows',
#       'percentage': args.percentage
#     }

#     model = AutoModelForCausalLM.from_pretrained(config["model_name"])
#     model_path = f"../bias_IF/opt_{config['data_type']}_350m_select_{config['val_type']}_bs{args.bs}/select_{config['data_type']}_small_model_{config['percentage']}_{config['val_type']}.pth"
#     # model_path = f"../bias_IF/opt_{config['data_type']}_350m_select_{config['val_type']}/select_{config['data_type']}_small_model_{config['percentage']}_{config['val_type']}.pth"

#     if config['model_path']:
#         model.load_state_dict(torch.load(model_path))

#     print(model_path)
    
#     model.eval()
#     tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
#     eos_token_id = tokenizer.eos_token_id
#     tokenizer.pad_token_id = [str(eos_token_id)]
#     runner = Runner(model, tokenizer, config['data_path'])
#     results = runner()

#     if config['model_path']:
#       output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows_ft.json"
#     else:
#       output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows.json"
       
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)

# import os
# import json
# import transformers
# import torch

# from val.CrowsRunner import Runner
# from transformers import AutoModelForCausalLM

# if __name__ == "__main__":
#     config = {
#       'model_name': 'facebook/opt-1.3b',
#       # 'model_path': r'./model',
#       'model_path': None,
#       'store_path': r"./output/crows",
#       'data_path': r'./data/crows/crows_pairs_anonymized.csv'
#     }

#     # model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict=True)
#     model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
#     if config['model_path']:
#         model.load_state_dict(torch.load(config['model_path']+"\\opt_mid_model.pth"))
#     model.eval()
#     tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
#     eos_token_id = tokenizer.eos_token_id
#     tokenizer.pad_token_id = [str(eos_token_id)]
#     runner = Runner(model, tokenizer, config['data_path'])
#     results = runner()

#     if config['model_path']:
#       output_path = f"{config['store_path']}/results/crows_ft.json"
#     else:
#       output_path = f"{config['store_path']}/results/crows.json"
       
#     # os.makedirs(output_path, exist_ok=True)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)
import os
import json
import transformers
import torch
import argparse

from val.CrowsRunner import Runner
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )
    
    # parser.add_argument(
    #     '--dataset_type',
    #     type=str,
    #     default='balance',
    #     help='Type of dataset to use'
    # )

    parser.add_argument(
        '--percentage',
        type=str,
        default='0.02',
        help='Percentage of data to use (default: 0.02)'
    )
    
    parser.add_argument(
        '--val_type',
        type=str,
        # choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'mix', 'news', 'toxic_mix'],
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
        '--model_name',
        type=str,
        default="facebook/opt-350m",
        choices=['Qwen/Qwen2.5-0.5B', 'facebook/opt-350m'],
        help='model_name'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.model_name == 'python -m experiments.crows_lora --val_type trex --model_name Qwen/Qwen2.5-1.5B-Instruct':
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
      # 'data_type': args.dataset_type,
    #   'val_type': 'crows',
      'val_type': args.val_type,
      'percentage': args.percentage
    }

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    if args.model_name == "facebook/opt-350m":
      model_path = f"../bias_sel/opt_{config['val_type']}_350m_select_full/select_{config['val_type']}_small_model.pth"
    else:
      model_path = f"../bias_sel/Qwen/Qwen_{config['val_type']}_500m_select_full/select_{config['val_type']}_small_model.pth"

    if config['model_path']:
        model.load_state_dict(torch.load(model_path))

    print(model_path)
    
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]
    runner = Runner(model, tokenizer, config['data_path'])
    results = runner()

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_{config['val_type']}/{config['percentage']}_bs{args.bs}/crows.json"
    if config['model_path']:
      output_path = f"{config['store_path']}/{name}/results_small_select_{config['val_type']}/crows_full.json"
    else:
      output_path = f"{config['store_path']}/{name}/results_small_select_{config['val_type']}/crows.json"
      
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/crows_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/crows.json"
       
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
