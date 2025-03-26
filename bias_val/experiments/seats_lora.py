import json
import os
import transformers
import argparse
import torch
import pdb
import json
import os
from statistics import mean

from val.SeatRunner import Runner
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModel, AutoModelForCausalLM

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
        default='0.05',
        help='Percentage of data to use (default: 0.02)'
    )
    
    parser.add_argument(
        '--val_type',
        type=str,
        default='pure_wino',
        # choices=['balance', 'wino', 'reddit', 'bug', 'pure_wino', 'chat_bias', 'stereoset', 'prompt', 'crows', 'seat', 'gen', 'pp', 'pt', 'prompt_few', 'mix', 'news', 'toxic_mix'],
        help='validation type: crows, stereoset, seat, wino, pure_wino, chat_bias, random'
    )

    parser.add_argument(
        '--dataset_percentage',
        type=str,
        default='few',
        help='few or full'
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
      'model_name': 'facebook/opt-1.3b',
    #   'tests': ['sent-weat6', 'sent-weat6b', 'sent-weat7', 'sent-weat7b', 'sent-weat8', 'sent-weat8b'],
      'tests': ['heilman_double_bind_likable_one_sentence', 'heilman_double_bind_competent_1+3-', 'heilman_double_bind_likable_one_sentence', 'sent-weat6', 'sent-weat6b'],
      'n_sample': 10000,
      'model_path': r'./model',
      # 'model_path': None,
      'store_path': r"./output/seat",
      'data_path': r'./data/seat',
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

    # adapter_model_dir = adapter_model_dir = f"../bias_IF/opt_{config['val_type']}_1.3b_select_ig/peft_model_{config['val_type']}_1.3b"
    # adapter_model_dir = adapter_model_dir = f"../bias_IF/opt_1.3b_peft_{config['data_type']}_select_{config['val_type']}/select_{config['percentage']}_peft_model_{config['data_type']}_{config['val_type']}"
    # adapter_model_dir = f"../bias_IF/opt_{config['val_type']}_1.3b_select_ig/select_{config['percentage']}_peft_model_{config['val_type']}"
    adapter_model_dir = f"../bias_sel/opt_{config['val_type']}_1.3b_select_full/peft_model_{config['val_type']}_1.3b"
  
    model_lm = AutoModelForCausalLM.from_pretrained(config['model_name'], return_dict=True)
    model = model_lm.model.decoder
    model = PeftModel(model, lora_config)
    print(adapter_model_dir)

    if config['model_path']:
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        model = PeftModel.from_pretrained(model, adapter_model_dir)
        # state_dict = torch.load(config['model_path']+c)
        # decoder_state_dict = {k.replace('base_model.model.model.decoder.', 'base_model.model.'): v for k, v in state_dict.items() if k.startswith('base_model.model.model.decoder.')}
        # model.load_state_dict(decoder_state_dict)
        model = model.model.model.decoder

    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_name'])
    eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = [str(eos_token_id)]

    runner = Runner(model, tokenizer, config['tests'], config['data_path'], config['n_sample'])
    results = runner()

    if config['model_path']:
      output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_{config['val_type']}/seats_ft.json"
    else:
      output_path = f"{config['store_path']}/results_mid_{config['data_type']}_select_{config['val_type']}/seats.json"
      
    # if config['model_path']:
    #   output_path = f"{config['store_path']}/results_mid_{config['val_type']}_select_seat_ig/seat_ft.json"
    # else:
    #   output_path = f"{config['store_path']}/results_mid_{config['val_type']}_select_seat_ig/seat.json"
       
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