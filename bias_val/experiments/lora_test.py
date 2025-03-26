from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType
import torch

config = {
    'model_name': 'facebook/opt-1.3b',
    'adapter_model_dir': './model/opt_1.3b_peft/peft_model_1.3b'
}

lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.CAUSAL_LM
    )

original_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
original_model = PeftModel(original_model, lora_config)

model_with_adapter = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_with_adapter = PeftModel.from_pretrained(model_with_adapter, config['adapter_model_dir'])

for (name_orig, param_orig), (name_adapter, param_adapter) in zip(
    original_model.named_parameters(), model_with_adapter.named_parameters()
):
    if name_orig == name_adapter:
        if not torch.equal(param_orig, param_adapter):
            print(f"参数 {name_orig} 不同")
            print("原始模型参数值示例:", param_orig.flatten()[:5])
            print("带适配器模型参数值示例:", param_adapter.flatten()[:5])
    else:
        print(f"参数名称不同：{name_orig} vs {name_adapter}")
