
import argparse
import os
import pdb
from copy import deepcopy
from typing import Any
from safetensors.torch import load_file
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel
from step2.collect_grad_reps import (collect_grads, collect_reps, get_loss)

class UCC_Dataset(Dataset):
  def __init__(self, data_path, tokenizer, max_token_len: int=128, sample=500):
    self.data_path = data_path
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self):
    self.data = pd.read_csv(self.data_path, header=0)

  def __len__(self):
    return (len(self.data))

  def __getitem__(self, index):
    sentence = str(self.data.iloc[index, 0])
    tokens = self.tokenizer.encode_plus(sentence,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=self.max_token_len,
                                        padding='max_length',
                                        return_attention_mask=True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten()}

class UCC_Data_Module(pl.LightningDataModule):

  # FIXME 这里改batch_size
  def __init__(self, data_path, batch_size: int=4, max_token_len: int=128, model_name = 'facebook/opt-1.3b'):
    super().__init__()
    self.data_path = data_path
    self.batch_size = batch_size
    self.max_token_len = max_token_len
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 

  def setup(self, stage = None):
    self.dataset = UCC_Dataset(self.data_path, self.tokenizer, max_token_len=self.max_token_len)
    # pdb.set_trace()

  def train_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def val_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
  
def load_and_map_optimizer_state(md_path, model):

    optimizer_path = os.path.join(md_path, "optimizer.bin")
    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]
    
    trainable_params = [name for name, param in model.named_parameters() 
                       if param.requires_grad and "lora" in name]
    
    if len(trainable_params) != len(adam_optimizer_state):
      print(f"Warning: Number of trainable parameters ({len(trainable_params)}) "
            f"doesn't match optimizer states ({len(adam_optimizer_state)})")
      print("\nTrainable parameter names:")
      pdb.set_trace()
      for name in trainable_params:
          print(f"- {name}")
      print("\nOptimizer state keys:")
      print(sorted(adam_optimizer_state.keys()))
      pdb.set_trace()
      return None
    
    mapped_state = {}
    
    for idx, key in enumerate(sorted(adam_optimizer_state.keys())):
        if idx >= len(trainable_params):
            print(f"Warning: More optimizer states than trainable parameters")
            break
        
        param_name = trainable_params[idx]
        state = adam_optimizer_state[key]
        
        if 'lora' in param_name:
            if 'exp_avg' in state:
                state['exp_avg'].requires_grad = True
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].requires_grad = True
        
        mapped_state[param_name] = state
        
    return mapped_state

parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument("--model_path", type=str, default="./", help="The path to the model")
parser.add_argument("--info_type", choices=["grads", "reps", "loss"], default="grads", help="The type of information")
# 改这里
parser.add_argument("--output_path", type=str, required=True, default='./step2/result_1.3b_crows_val', help="The path to the output")
parser.add_argument("--gradient_projection_dimension", nargs='+', help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--max_samples", type=int, default=None, help="The maximum number of samples")
parser.add_argument("--gradient_type", type=str, default="adam", required=True,
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--ds", type=str, required=True, default="val_data/prompt_data.csv",
                    help="ds")
parser.add_argument("--md_path", type=str, required=True, default="./opt_prompt_1.3b_select_ig",
                    help="model path")
parser.add_argument("--dataset_type", type=str, required=True, default="gen_mix_full")
parser.add_argument("--bs", type=int, default=4,
                    help="bs")
parser.add_argument(
    '--model_name',
    type=str,
    default='Qwen/Qwen2.5-1.5B-Instruct',
    required=True,
    choices=['Qwen/Qwen2.5-1.5B-Instruct', 'facebook/opt-1.3b'],
    help='model_name'
)

if __name__ == '__main__':
    args = parser.parse_args()
    
    model_name = args.model_name
    
    md_path = args.md_path
    pre_model_name = f'peft_model_{args.dataset_type}_1.3b' if model_name == 'facebook/opt-1.3b' else f'peft_model_{args.dataset_type}_mid'
    
    dtset = args.ds
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)
    model = PeftModel.from_pretrained(model, os.path.join(md_path, pre_model_name))
    model.train()
    model.to(device)
    for name, param in model.named_parameters():
      if "lora" in name:
        param.requires_grad = True
        
    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    adam_optimizer_state = load_and_map_optimizer_state(md_path, model)

    ucc_data_module = UCC_Data_Module(os.path.join("./data", dtset), model_name = model_name, batch_size=args.bs)
    ucc_data_module.setup()
    dl = ucc_data_module.train_dataloader()

    if args.info_type == "reps":
        collect_reps(dl, model, args.output_path,
                    max_samples=args.max_samples)
    elif args.info_type == "grads":
        collect_grads(dl,
                    model,
                    args.output_path,
                    proj_dim=args.gradient_projection_dimension,
                    gradient_type=args.gradient_type,
                    adam_optimizer_state=adam_optimizer_state,
                    max_samples=args.max_samples)
    elif args.info_type == "loss":
        get_loss(dl, model, args.output_path)
