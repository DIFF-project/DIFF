# import argparse
# import os
# import pdb
# from copy import deepcopy
# from typing import Any
# from safetensors.torch import load_file
# import torch
# import pytorch_lightning as pl
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
# import torch.nn as nn

# from step2.collect_grad_reps import (collect_grads, collect_reps, get_loss)

# class UCC_Dataset(Dataset):
#   def __init__(self, data_path, tokenizer, max_token_len: int=128, sample=500):
#     self.data_path = data_path
#     self.tokenizer = tokenizer
#     self.max_token_len = max_token_len
#     self.sample = sample
#     self._prepare_data()

#   def _prepare_data(self):
#     self.data = pd.read_csv(self.data_path, header=0)

#   def __len__(self):
#     return (len(self.data))

#   def __getitem__(self, index):
#     sentence = str(self.data.iloc[index, 0])
#     tokens = self.tokenizer.encode_plus(sentence,
#                                         add_special_tokens=True,
#                                         return_tensors='pt',
#                                         truncation=True,
#                                         max_length=self.max_token_len,
#                                         padding='max_length',
#                                         return_attention_mask=True)
#     return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten()}

# class UCC_Data_Module(pl.LightningDataModule):

#   # FIXME 这里改batch_size
#   def __init__(self, data_path, batch_size: int=1, max_token_len: int=128, model_name = 'facebook/opt-1.3b'):
#     super().__init__()
#     self.data_path = data_path
#     self.batch_size = batch_size
#     self.max_token_len = max_token_len
#     self.model_name = model_name
#     self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     eos_token_id = self.tokenizer.eos_token_id
#     self.tokenizer.pad_token_id = [str(eos_token_id)]

#   def setup(self, stage = None):
#     self.dataset = UCC_Dataset(self.data_path, self.tokenizer, max_token_len=self.max_token_len)
#     # pdb.set_trace()

#   def train_dataloader(self):
#     return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

#   def val_dataloader(self):
#     return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

#   def predict_dataloader(self):
#     return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


# # def load_model(model_path: str) -> Any:

# #     model = AutoModelForCausalLM.from_pretrahined("gpt2", return_dict=True)
# #     loaded_state_dict = load_file(model_path+"\\model.safetensors")
# #     model.load_state_dict(loaded_state_dict)
# #     model.eval()
# #     # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

# #     for name, param in model.named_parameters():
# #         if 'lora' in name or 'Lora' in name:
# #             param.requires_grad = True
# #     return model


# parser = argparse.ArgumentParser(
#     description='Script for getting validation gradients')
# parser.add_argument("--model_path", type=str, default="./", help="The path to the model")
# parser.add_argument("--info_type", choices=["grads", "reps", "loss"], default="grads", help="The type of information")
# # 改这里
# parser.add_argument("--output_path", type=str, default='./step2/result', help="The path to the output")
# parser.add_argument("--gradient_projection_dimension", nargs='+', help="The dimension of the projection, can be a list", type=int, default=[8192])
# parser.add_argument("--max_samples", type=int, default=None, help="The maximum number of samples")
# parser.add_argument("--gradient_type", type=str, default="adam",
#                     choices=["adam", "sign", "sgd"], help="The type of gradient")

# if __name__ == '__main__':
#     args = parser.parse_args()
#     tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b', padding=True, truncation=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # model = load_model(args.model_path)
#     model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", return_dict=True)
#     # loaded_state_dict = load_file(".\\model.pth")
#     # model.load_state_dict(loaded_state_dict)
#     model.load_state_dict(torch.load('./opt_3b/opt_mid_model.pth'))
#     # model = nn.DataParallel(model)
#     model.to(device)

#     if tokenizer.pad_token is None:
#         eos_token_id = tokenizer.eos_token_id
#         tokenizer.pad_token_id = [str(eos_token_id)]

#     embedding_size = model.get_input_embeddings().weight.shape[0]
#     if len(tokenizer) > embedding_size:
#         model.resize_token_embeddings(len(tokenizer))

#     adam_optimizer_state = None
#     optimizer_path = os.path.join("./opt_3b", "optimizer.bin")
#     adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]
#     # print("Keys in 'state' dictionary:")
#     # for key in adam_optimizer_state.keys():
#     #     print(f"Key: {key}")
#     # print("\nContent of each key in 'state' dictionary:")
#     # for param_key, param_state in adam_optimizer_state.items():
#     #     print(f"\nParameter key: {param_key}")
#     #     for state_key, state_value in param_state.items():
#     #         print(f"  State key: {state_key}, Type: {type(state_value)}")
#     #         if isinstance(state_value, torch.Tensor):
#     #             print(f"  State value shape: {state_value.shape}")
#     # pdb.set_trace()

#     ucc_data_module = UCC_Data_Module(os.path.join("./data", "balanced_dataset.csv"))
#     # ucc_data_module = UCC_Data_Module(os.path.join("./data", "val_sent.csv"))
#     ucc_data_module.setup()
#     dl = ucc_data_module.train_dataloader()
#     # dl = ucc_data_module.val_dataloader()
    

#     if args.info_type == "reps":
#         collect_reps(dl, model, args.output_path,
#                     max_samples=args.max_samples)
#     elif args.info_type == "grads":
#         collect_grads(dl,
#                     model,
#                     args.output_path,
#                     # proj_dim=args.gradient_projection_dimension,
#                     gradient_type=args.gradient_type,
#                     adam_optimizer_state=adam_optimizer_state,
#                     max_samples=args.max_samples)
#     elif args.info_type == "loss":
#         get_loss(dl, model, args.output_path)

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
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from transformers import OPTForCausalLM

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
    
    # model_name = 'facebook/opt-1.3b'
    model_name = 'facebook/opt-350m'
    
    # md_path = './opt_1.3b_peft'
    # md_path = './opt_wino_1.3b_peft'
    md_path = args.md_path
    # md_path = './opt_prompt_350m_select_ig'
    # md_path = './opt_balance_350m_select'
    
    # dtset = "balanced_dataset.csv" 
    dtset = args.ds 
    # dtset = "val_sent.csv"
    
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
    # print("Keys in 'state' dictionary:")
    # for key in adam_optimizer_state.keys():
    #     print(f"Key: {key}")
    # print("\nContent of each key in 'state' dictionary:")
    # for param_key, param_state in adam_optimizer_state.items():
    #     print(f"\nParameter key: {param_key}")
    #     for state_key, state_value in param_state.items():
    #         print(f"  State key: {state_key}, Type: {type(state_value)}")
    #         if isinstance(state_value, torch.Tensor):
    #             print(f"  State value shape: {state_value.shape}")

    ucc_data_module = UCC_Data_Module(os.path.join("./data", dtset), model_name = model_name, batch_size=args.bs)
    # ucc_data_module = UCC_Data_Module(os.path.join("./data", "val_sent.csv"))
    ucc_data_module.setup()
    dl = ucc_data_module.train_dataloader()
    # dl = ucc_data_module.val_dataloader()
    

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
