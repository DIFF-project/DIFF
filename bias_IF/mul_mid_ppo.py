import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from datasets import Dataset
from trl import PPOTrainer, PPOConfig
from googleapiclient import discovery
from transformers import pipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration for validation and dataset selection'
    )

    parser.add_argument(
        '--val_type',
        type=str,
        default='few',
        help='Type of validation to perform'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='gen_mix',
        help='Type of dataset to use'
    )
    
    parser.add_argument(
        '--bs',
        type=int,
        default=16,
        help='batch size'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='epoch'
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

def reward_function(generated_text, label):
    if label == generated_text:
        reward = 1.0
    else:
        reward = -1.0
    
    return reward

if __name__ == '__main__':

    args = parse_args()
    config = {
      'model_name': args.model_name,
      'bs': args.bs,
      'lr': 1.5e-6,
      'warmup': 0.2,
      'w_decay': 0.001,
    }
    if args.model_name == 'Qwen/Qwen2.5-1.5B-Instruct':
        name = 'Qwen'
    else:
        name = 'opt'

    od = f'./{name}/{name}_{args.dataset_type}_mid_select_ppo'

    if args.model_name == "facebook/opt-1.3b":
        adapter_model_dir = f"./opt_{config['val_type']}_1.3b_select_full/peft_model_{config['val_type']}_1.3b"
    else:
        adapter_model_dir = f"./Qwen/Qwen_{config['val_type']}_1.5b_select_full/peft_model_{config['val_type']}_mid"

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(model, adapter_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    data_path = f"./steprl/{name}/{name}_{args.dataset_type}_mid.csv"
    data = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(data)

    ppo_config = PPOConfig(
        batch_size=args.bs,
        learning_rate=1.5e-5,
        ppo_epochs=40,
    )

    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer, dataset=dataset)
    prompt_text =  "Read the following sentence and decide whether it is a stereotype or an antistereotype. You should only output either 'stereo' or 'antistereo"

    for epoch in range(args.epoch):
        for batch in ppo_trainer.dataloader:
            input_texts = batch["sent"]
            labels = batch["stereo_antistereo"]
            input_texts = [prompt_text + " " + text for text in input_texts]
            
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True)

            outputs = model.generate(**inputs, max_length=50)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            generated_labels = []
            for text in generated_texts:
                if "antistereo" in text.lower():
                    generated_labels.append("antistereoset")
                elif "stereo" in text.lower():
                    generated_labels.append("stereoset")
                else:
                    generated_labels.append(None)

            rewards = [reward_function(text, label) for text, label in zip(generated_labels, labels)]
            
            ppo_trainer.step(inputs["input_ids"], outputs, rewards)

    model.save_pretrained(od)
    tokenizer.save_pretrained(od)