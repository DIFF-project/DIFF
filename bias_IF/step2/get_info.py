import argparse
import os
import pdb

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from step2.collect_grad_reps import collect_grads, collect_reps, get_loss

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
  # train 的batchsize目前都是32
  # val 改成2
  def __init__(self, data_path, batch_size: int=4, max_token_len: int=128, model_name = 'facebook/opt-350m'):
    super().__init__()
    self.data_path = data_path
    self.batch_size = batch_size
    self.max_token_len = max_token_len
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = self.tokenizer.eos_token_id
    self.tokenizer.pad_token_id = [str(eos_token_id)]

  def setup(self, stage = None):
    self.dataset = UCC_Dataset(self.data_path, self.tokenizer, max_token_len=self.max_token_len)
  def train_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def val_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

parser = argparse.ArgumentParser(
    description='Script for getting validation gradients')
parser.add_argument("--model_path", type=str, default="./", help="The path to the model")
parser.add_argument("--info_type", choices=["grads", "reps", "loss"], default="grads", help="The type of information")
parser.add_argument("--output_path", type=str, default='./step2/result_350m_toxic_mix_val_full', help="The path to the output")
parser.add_argument("--gradient_projection_dimension", nargs='+', help="The dimension of the projection, can be a list", type=int, default=[8192])
parser.add_argument("--max_samples", type=int, default=None, help="The maximum number of samples")
parser.add_argument("--gradient_type", type=str, default="adam",
                    choices=["adam", "sign", "sgd"], help="The type of gradient")
parser.add_argument("--ds", type=str, default="val_data/prompt_data.csv",
                    help="ds")
parser.add_argument("--bs", type=int, default=4,
                    help="bs")  
parser.add_argument("--dataset_type", type=str, default="gen_mix_few")
parser.add_argument("--md_path", type=str, default="./opt_prompt_few_350m_select_ig",
                    help="md") 

if __name__ == '__main__':
    args = parser.parse_args()
    
    model_name = 'facebook/opt-350m'
    md_path = args.md_path
    dtset = args.ds 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)
    # model.load_state_dict(torch.load(os.path.join(md_path, 'peft_model_1.3b.pth')))
    model.load_state_dict(torch.load(os.path.join(md_path, f'select_{args.dataset_type}_small_model.pth')))
    model.to(device)

    if tokenizer.pad_token is None:
        eos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = [str(eos_token_id)]

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    adam_optimizer_state = None
    optimizer_path = os.path.join(md_path, "optimizer.bin")
    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]

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
