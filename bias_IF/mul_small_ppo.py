import pandas as pd
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import Dataset
from trl import PPOTrainer, PPOConfig
from googleapiclient import discovery
from transformers import pipeline
import argparse

# API_KEY = 'AIzaSyCk6LQap2W6lX0RQJQ1EKM27cpibH6bPNc'

# client = discovery.build(
#   "commentanalyzer",
#   "v1alpha1",
#   developerKey=API_KEY,
#   discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#   static_discovery=False,
# )

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
        default="facebook/opt-350m",
        choices=['Qwen/Qwen2.5-0.5B', 'facebook/opt-350m'],
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

# def reward_function(generated_text):
#     toxic_score = compute_toxic_reward(generated_text)
#     fluency_result = fluency_scorer(generated_text)[0]
#     fluency_score = fluency_result["score"] if fluency_result["label"] == "FLUENT" else 1 - fluency_result["score"]
    
#     toxic_weight = 0.8
#     fluency_weight = 0.2 
    
#     reward = (toxic_weight * toxic_score) + (fluency_weight * fluency_score)
#     return reward

# def compute_toxic_reward(text):
#     analyze_request = {
#         'comment': { 'text': text },
#         'requestedAttributes': {'TOXICITY': {}}
#     }

#     response = client.comments().analyze(body=analyze_request).execute()
#     toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
#     toxic_reward = 1 - toxicity
#     return toxic_reward

if __name__ == '__main__':

    args = parse_args()

    if args.model_name == 'Qwen/Qwen2.5-0.5B':
        name = 'Qwen'
    else:
        name = 'opt'

    od = f'./{name}/{name}_{dataset_type}_small_select_ppo'

    # fluency_scorer = pipeline("text-classification", model="text-fluency-model")
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    if args.model_name == "facebook/opt-350m":
        model_path = f"./opt_{config['val_type']}_350m_select_full/select_{config['val_type']}_small_model.pth"
    else:
        model_path = f"./Qwen/Qwen_{config['val_type']}_500m_select_full/select_{config['val_type']}_small_model.pth"

    if config['model_path']:
        model.load_state_dict(torch.load(model_path))

    data_path = f"./steprl/{name}/{name}_{dataset_type}_small.csv"
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