import json
import os
import transformers
import torch
import argparse
import pdb
import json
import os
from statistics import mean

from val.SeatRunner import Runner
from transformers import AutoModel, AutoModelForCausalLM

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
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = {
      'model_name': 'facebook/opt-350m',
      'tests': ['heilman_double_bind_likable_one_sentence', 'heilman_double_bind_competent_1+3-', 'heilman_double_bind_likable_one_sentence', 'sent-weat6', 'sent-weat6b'],
      'n_sample': 10000,
      'model_path': r'./model',
    #   'model_path': None,
      'store_path': r"./output/seat",
      'data_path': r'./data/seat',
      'data_type': args.dataset_type,
      'val_type': args.val_type,
      'percentage': args.percentage
    }

    model_lm = AutoModelForCausalLM.from_pretrained(config['model_name'], return_dict=True)
    model = model_lm.model.decoder

    if config['model_path']:
        # model_path = f"../bias_IF/opt_{config['data_type']}_350m_select_{config['val_type']}/select_{config['data_type']}_small_model_{config['percentage']}_{config['val_type']}.pth"
        # model_path = f"../bias_IF/opt_{config['val_type']}_350m_select_ig/select_{config['val_type']}_small_model.pth"
        model_path = f"../bias_sel/opt_{config['val_type']}_350m_select_full/select_{config['val_type']}_small_model.pth"
        state_dict = torch.load(model_path)
        decoder_state_dict = {k.replace('model.decoder.', ''): v for k, v in state_dict.items() if k.startswith('model.decoder.')}
        model.load_state_dict(decoder_state_dict)

    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]

    runner = Runner(model, tokenizer, config['tests'], config['data_path'], config['n_sample'])
    results = runner()

    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_seat_{config['val_type']}/{config['percentage']}/seats_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['data_type']}_select_seat_{config['val_type']}/{config['percentage']}/seats.json"
      
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/seats_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_small_{config['val_type']}_select_ig/seats.json"
      
    if config['model_path']:
      output_path = f"{config['store_path']}/results_small_select_{config['val_type']}/seat_full.json"
    else:
      output_path = f"{config['store_path']}/results_small_select_{config['val_type']}/seat.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)

    file_path = output_path

    with open(file_path, 'r') as file:
        data = json.load(file)

    # 提取p_value和effect_size
    p_values = [item['p_value'] for item in data]
    effect_sizes = [item['effect_size'] for item in data]

    # 计算平均值
    avg_p_value = mean(p_values)
    avg_effect_size = mean(effect_sizes)

    # 打印结果
    print(f"p_value 的平均值：{avg_p_value:.4f}")
    print(f"effect_size 的平均值：{avg_effect_size:.4f}")