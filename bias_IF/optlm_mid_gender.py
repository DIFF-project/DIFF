import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModel, AdamW, GPT2LMHeadModel, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import torch.nn as nn
import math
import warnings
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OPTForCausalLM
import pdb
import json
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from pytorch_lightning.callbacks import ModelCheckpoint

# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

  def __init__(self, data_path, batch_size: int=16, max_token_len: int=128, model_name = "facebook/opt-350m"):
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
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

class UCC_Classifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.val_losses = []
        self.train_losses = []
        self.validation_step_outputs = []
        self.best_val_loss = float('inf')
        self.config = config
        self.pretrained_model = OPTForCausalLM.from_pretrained(config['model_name'], return_dict=True)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=2)
        self.dropout = nn.Dropout()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = [str(eos_token_id)]
    
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        self.pretrained_model = get_peft_model(self.pretrained_model, lora_config)
        self.pretrained_model.train()
        self.pretrained_model.print_trainable_parameters()
    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # loss = outputs.loss
        # pdb.set_trace()
        labels = input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()
        loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, logits

    def training_step(self, batch, batch_index):
        # if self.current_epoch > 0:
        #     pdb.set_trace()
        loss, logits = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # if batch_index % 100 == 0:
        self.train_losses.append(loss.item())
        return {"loss": loss, "predictions": logits}

    def validation_step(self, batch, batch_index):
        loss, logits = self(**batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": logits}

    def predict_step(self, batch, batch_index):
        _, logits = self(**batch)
        return logits
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.val_losses.append(avg_loss.item())
        print(f"Current avg_loss is {avg_loss}")
        # if avg_loss < self.best_val_loss:
        self.best_val_loss = avg_loss
        output_dir = od
        self.save_model(output_dir)
        self.save_optimizer_state(output_dir) 
            
        self.validation_step_outputs.clear()
        output_dir = './fig'
        self.validation_step_outputs = []

        # plt.figure(figsize=(12, 6))
        # plt.plot(self.train_losses, label='Training Loss')
        # plt.plot(self.val_losses, label='Validation Loss')
        # plt.xlabel('Steps')
        # plt.ylabel('Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.grid(True)
        # loss_plot_filename = os.path.join(output_dir, f'loss_plot_epoch_{self.current_epoch}_mix_few_train.png')
        # plt.savefig(loss_plot_filename)
        # plt.close()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['bs']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # adapter_path = os.path.join(output_dir, f"select_{percentage}_peft_model_{dataset_type}")
        # adapter_path = os.path.join(output_dir, f"select_{percentage}_peft_model_{dataset_type}_{val_type}")
        adapter_path = os.path.join(output_dir, f"peft_model_{dataset_type}_1.3b")
        self.pretrained_model.peft_config["default"].inference_mode = False
        self.pretrained_model.save_pretrained(adapter_path)

    def save_optimizer_state(self, output_dir):
        optimizer_path = os.path.join(output_dir, "optimizer.bin")
        torch.save(self.optimizers().state_dict(), optimizer_path)

    def load_optimizer_state(self, optimizer, optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )

    parser.add_argument(
        '--val_type',
        type=str,
        default='crows',
        # choices=['crows', 'stereoset', 'seat'],
        help='Type of validation to perform'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='balance',
        # choices=['balance', 'wino', 'pure_wino', 'chat_bias', 'stereoset', 'crows', 'seat', 'prompt', 'gen', 'prompt_few', 'mix', 'mix_few', 'news', 'toxic_mix', 'toxic_mix_few'],
        help='Type of dataset to use'
    )

    parser.add_argument(
        '--percentage',
        type=float,
        default=0.02,
        help='Percentage of data to use (default: 0.02)'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=8,
        help='batch size'
    )
    
    args = parser.parse_args()
    return args

args = parse_args()
val_type = args.val_type
percentage = args.percentage
dataset_type = args.dataset_type
# od = f'./opt_1.3b_peft_{dataset_type}_select_{val_type}_bs{args.bs}'
# od = f'./opt_1.3b_peft_{dataset_type}_select_{val_type}'
if dataset_type in ['stereoset', 'crows', 'seat', 'prompt', 'gen', 'mix', 'news', 'toxic_mix', 'crows_t', 'gen_mix']:
    od = f'./opt_{dataset_type}_1.3b_select_full'
else:
    od = f'./opt_{dataset_type}_1.3b_select_ig'


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./opt_{dataset_type}_1.3b_select_full", 
        filename=f'{dataset_type}_bestmodel',
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    if not os.path.exists(od):
        os.makedirs(od)
        print(f"创建目录：{od}")
    else:
        print(f"目录已存在：{od}")
        
    # data_path = f"./data/{dataset_type}_random/{percentage}/{dataset_type}_random_sample.csv"
    if dataset_type == 'pure_wino' or dataset_type == 'chat_bias':
        data_path = f"./data/{dataset_type}.csv"
    if dataset_type in ['stereoset', 'crows', 'seat', 'prompt', 'gen', 'mix', 'news', 'toxic_mix', 'crows_t', 'gen_mix']:
        data_path = f"./data/val_data/{dataset_type}_data.csv"
    if dataset_type in ['prompt_few', 'mix_few', 'toxic_mix_few', 'gen_mix_few']:
        mapping = {
            'prompt_few': 'prompt',
            'mix_few': 'mix',
            'toxic_mix_few': 'toxic_mix',
            'gen_mix_few': 'gen_mix'
        }
        data_path = f"./data/{mapping[dataset_type]}_random/{percentage}/{mapping[dataset_type]}_random_sample.csv"
    if dataset_type == 'crows_gen':
        data_path = "./data/step2_val_data/crows_gen_data.csv"
    if dataset_type[-4:] == 'full':
        data_path = f"./data/val_data/{dataset_type[:-5]}_data.csv"
    
    model_name = 'facebook/opt-1.3b'
    ucc_data_module = UCC_Data_Module(data_path)
    ucc_data_module.setup()
    print(data_path)

    dl = ucc_data_module.train_dataloader()

    config = {
      'model_name': model_name,
      'bs': args.bs,
      'lr': 1.5e-6,
      'warmup': 0.2,
      'train_size': len(ucc_data_module.train_dataloader()),
      'w_decay': 0.001,
      'n_epochs': 50 if dataset_type in ['stereoset', 'crows', 'seat', 'prompt', 'gen', 'mix', 'news', 'toxic_mix', 'crows_t', 'gen_mix'] else 5
    }

    ucc_data_module = UCC_Data_Module(data_path, batch_size=config['bs'], model_name=model_name)
    ucc_data_module.setup()

    model = UCC_Classifier(config)

    trainer = pl.Trainer(max_epochs=config['n_epochs'], devices=[0], num_sanity_val_steps=10, callbacks=[checkpoint_callback])
    trainer.fit(model, ucc_data_module)
    # model.save_optimizer_state(output_dir)