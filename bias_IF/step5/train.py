from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from utils.custom_dataset import CustomDataset
from utils.loss import *
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset_model_name', type=str, default="Llama-3-8B-Instruct")
    parser.add_argument('--output_dir', type=str, default="output_model")
    parser.add_argument('--type', type=int, default=1)

    # 用来找从哪个ckpt开始
    parser.add_argument(
        '--val_type',
        type=str,
        default='full',
        help='Type of validation to perform'
    )

    parser.add_argument(
        '--dataset_type',
        type=str,
        default='balance',
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
        default=4,
        help='batch_size'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="facebook/opt-1.3b",
        choices=['Qwen/Qwen2.5-1.5B-Instruct', 'facebook/opt-1.3b', 'Qwen/Qwen2.5-0.5B', 'facebook/opt-350m'],
        help='model_name'
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.model_name in ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B', 'facebook/opt-350m']:
        name = 'Qwen'
        if args.model_name == 'Qwen/Qwen2.5-0.5B':
            num = '500m'
        else:
            num = '1.5b'
    else:
        name = 'opt'
        if args.model_name == 'facebook/opt-1.3b':
            num = '1.3b'
        else:
            num = '350m'

    od = f'../{name}/npo/{args.type}_{name}_{args.dataset_type}_{num}_select_npo_{args.val_type}'

    if args.model_name == "facebook/opt-350m":
        model_size = '350m'
        config = {
            'model_name': args.model_name,
            'model_path': f'../opt_{args.dataset_type}_350m_select_retrained_{args.val_type}/select_{args.dataset_type}_small_model_{args.percentage}_{args.val_type}.pth',
            # 'model_path': f'../bias_sel/opt_{args.dataset_type}_350m_select_full/select_{args.val_type}_small_model.pth',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': model_size
        }

        model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
        model.load_state_dict(torch.load(config['model_path']))

        if args.type in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14]:
            ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
            ref_model.load_state_dict(torch.load(config['model_path']))
            ref_model.eval()

    elif args.model_name == 'Qwen/Qwen2.5-0.5B':
        model_size = '500m'
        config = {
            'model_name': args.model_name,
            'model_path': f"../Qwen/Qwen_{args.dataset_type}_500m_select_full/select_{args.dataset_type}_small_model.pth",
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': model_size
        }

        model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
        model.load_state_dict(torch.load(config['model_path']))

        if args.type in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14]:
            ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
            ref_model.load_state_dict(torch.load(config['model_path']))
            ref_model.eval()


    elif args.model_name == 'Qwen/Qwen2.5-1.5B-Instruct':
        model_size = '1.5b'
        config = {
            'model_name': args.model_name,
            'model_path': f"../Qwen/Qwen_{args.dataset_type}_1.5b_select_full/peft_model_{args.dataset_type}_mid",
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': model_size
        }

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
        # model = PeftModel(model, lora_config)
        model = PeftModel.from_pretrained(model, config["model_path"])
        model.train()
        for nm, param in model.named_parameters():
            if 'lora' in nm or 'Lora' in nm:
                param.requires_grad = True
        # model = model.merge_and_unload()

        if args.type in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14]:
            ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
            # ref_model = PeftModel(ref_model, lora_config)
            ref_model = PeftModel.from_pretrained(ref_model, config["model_path"])
            ref_model = ref_model.merge_and_unload()
            ref_model.eval()

    # TODO opt加载模型还没改
    else:
        model_size = '1.3b'
        config = {
            'model_name': args.model_name,
            # 'model_path': f'../bias_sel/opt_{args.dataset_type}_1.3b_select_retrained_{args.val_type}/select_{args.percentage}_peft_model_{args.dataset_type}_{args.val_type}',
            'model_path': f'../opt_{args.dataset_type}_1.3b_select_retrained_{args.val_type}/select_{args.percentage}_peft_model_{args.dataset_type}_{args.val_type}',
            'data_type': args.dataset_type,
            'val_type': args.val_type,
            'percentage': args.percentage,
            'model_size': model_size
        }

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel(model, lora_config)
        model = PeftModel.from_pretrained(model, config["model_path"])
        model = model.merge_and_unload()

        if args.type in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14]:
            ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto")
            ref_model = PeftModel(ref_model, lora_config)
            ref_model = PeftModel.from_pretrained(ref_model, config["model_path"])
            ref_model = model.merge_and_unload()
            ref_model.eval()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # For W-DOOR, load dpo_model
    if args.type in [11, 12]:
        if "gemma" in args.model_name:
            dpo_name = "./output_model/gemma-2-2b-it/9"
        else:
            dpo_name = "./output_model/Llama-3-8B-Instruct/9"
        dpo_model = AutoModelForCausalLM.from_pretrained(
            dpo_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        dpo_model.eval()

    # Dataset dictionary
    # Not augmented
    if args.type in [1, 2, 3, 4, 5]:
        if args.type == 1:
            train_dataset_dict = {
                "": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 2:
            train_dataset_dict = {
                "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 3:
            train_dataset_dict = {
                "dpo_preferred_": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
                "dpo_non_preferred_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 4:
            train_dataset_dict = {
                "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 5:
            train_dataset_dict = {
                "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
                "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
    # Augmented
    elif args.type in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if args.type == 6:
            train_dataset_dict = {
                "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 7:
            train_dataset_dict = {
                "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 8:
            train_dataset_dict = {
                "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 9:
            train_dataset_dict = {
                "dpo_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
                "dpo_non_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type == 10:
            train_dataset_dict = {
                "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
                "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type in [11, 12]:
            train_dataset_dict = {
                "dpo_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
                "dpo_non_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
                "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
            }
        elif args.type in [13, 14]:
            train_dataset_dict = {
                "npo_": (f"../step5/{name}/{num}_{args.dataset_type}_{args.val_type}/{args.percentage}/bias_sent.jsonl", False),
                "utility_": (f"../step5/{name}/{num}_{args.dataset_type}_{args.val_type}/{args.percentage}/util_sent.jsonl", False)
            }

    dataset = CustomDataset(tokenizer, train_dataset_dict, max_words=512)

    def data_collator(batch_list):
        result = {}
        for key in batch_list[0].keys():
            if isinstance(batch_list[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch_list])
            else:
                raise ValueError(f"Unsupported type for key {key}")

        # print("Batch Input Data:", result)
        return result

    # import pdb
    # pdb.set_trace()
    training_arguments = TrainingArguments(
        output_dir=f"{od}/{args.type}-checkpoint",
        num_train_epochs=10.0,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=3e-7,
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="epoch",
        remove_unused_columns=False,
        save_only_model=True
    )

    # Trainer
    if args.type == 1:
        trainer = GATrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 2:
        trainer = NPOTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 3:
        trainer = GDTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 4:
        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 5:
        trainer = DOORTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 6:
        trainer = GATrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 7:
        trainer = NPOTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 8:
        trainer = GDTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 9:
        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 10:
        trainer = DOORTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 11:
        trainer = WDOORSIGTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            dpo_model=dpo_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
            gamma=1.0
        )
    elif args.type == 12:
        trainer = WDOORTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            dpo_model=dpo_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
            tau=1.0
        )
    elif args.type == 13:
        trainer = UnpairDOORTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif args.type == 14:
        trainer = ParetoDOORTrainer(
            model=model,
            train_dataset=dataset,
            args=training_arguments,
            ref_model=ref_model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    trainer.train()
    # final_output_dir = f"{args.output_dir}/{args.dataset_model_name}/{args.type}"
    trainer.save_model(od)