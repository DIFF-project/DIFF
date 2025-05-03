from transformers import Trainer, TrainerState
from transformers.utils import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from min_norm_solvers import MinNormSolver

import collections
import inspect
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

logger = logging.get_logger(__name__)
# Utility functions
def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1, -2), shifted_labels)
    return loss

def get_sequence_log_probs(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    log_probs = -loss_function(logits.transpose(-1, -2), shifted_labels)
    sequence_log_probs = log_probs.sum(dim=-1)
    return sequence_log_probs

def get_log_probs(output, labels):
    return -get_batch_loss(output, labels)

# Loss functions
def dpo_loss(main_model, ref_model, inputs, beta=0.5, retain_loss=True, alpha=0.2):
    device = next(main_model.parameters()).device
    preferred_input_ids = inputs['dpo_preferred_input_ids'].to(device)
    preferred_labels = inputs['dpo_preferred_labels'].to(device)
    preferred_attention_mask = inputs['dpo_preferred_attention_mask'].to(device)
    non_preferred_input_ids = inputs['dpo_non_preferred_input_ids'].to(device)
    non_preferred_labels = inputs['dpo_non_preferred_labels'].to(device)
    non_preferred_attention_mask = inputs['dpo_non_preferred_attention_mask'].to(device)

    preferred_outputs_main = main_model(preferred_input_ids, attention_mask=preferred_attention_mask)
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        preferred_outputs_ref = ref_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    preferred_log_probs_main = get_sequence_log_probs(preferred_outputs_main.logits, preferred_labels)
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    preferred_log_probs_ref = get_sequence_log_probs(preferred_outputs_ref.logits, preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    loss_val = -F.logsigmoid(beta * ((preferred_log_probs_main - preferred_log_probs_ref) -
                                     (non_preferred_log_probs_main - non_preferred_log_probs_ref))) * 2 / beta
    loss_val = loss_val.mean()

    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val

def npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2, retain_loss=True):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)

    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    loss_val = loss_val.mean()

    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val

def gd_loss(model, inputs, alpha=0.2):
    safety_input_ids = inputs['input_ids'].to(model.device)
    safety_labels = inputs['labels'].to(model.device)
    safety_attention_mask = inputs['attention_mask'].to(model.device)

    utility_input_ids = inputs['utility_input_ids'].to(model.device)
    utility_labels = inputs['utility_labels'].to(model.device)
    utility_attention_mask = inputs['utility_attention_mask'].to(model.device)

    safety_outputs = model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = get_batch_loss(safety_logits, safety_labels)

    utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_logits = utility_outputs.logits
    utility_loss = get_batch_loss(utility_logits, utility_labels)

    total_loss = alpha * torch.mean(safety_loss) + (1 - alpha) * torch.mean(utility_loss)
    return total_loss

def ga_loss(model, inputs, alpha=0.2):
    safety_input_ids = inputs['input_ids'].to(model.device)
    safety_labels = inputs['labels'].to(model.device)
    safety_attention_mask = inputs['attention_mask'].to(model.device)
    utility_input_ids = inputs['utility_input_ids'].to(model.device)
    utility_labels = inputs['utility_labels'].to(model.device)
    utility_attention_mask = inputs['utility_attention_mask'].to(model.device)
    safety_outputs = model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = -get_batch_loss(safety_logits, safety_labels)
    utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_logits = utility_outputs.logits
    utility_loss = get_batch_loss(utility_logits, utility_labels)
    total_loss = alpha * torch.mean(safety_loss) + (1 - alpha) * torch.mean(utility_loss)
    return total_loss

def gd_npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)
    
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)
    
    npo_loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    npo_loss_val = npo_loss_val.mean()
    
    safety_input_ids = inputs['input_ids'].to(device)
    safety_labels = inputs['labels'].to(device)
    safety_attention_mask = inputs['attention_mask'].to(device)
    safety_outputs = main_model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = -get_sequence_log_probs(safety_logits, safety_labels)
    safety_loss = safety_loss.mean()
    
    utility_input_ids = inputs['utility_input_ids'].to(device)
    utility_labels = inputs['utility_labels'].to(device)
    utility_attention_mask = inputs['utility_attention_mask'].to(device)
    utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
    utility_loss = -utility_log_probs.mean()
    
    total_loss = alpha * (safety_loss + npo_loss_val) + (1 - alpha) * utility_loss
    return total_loss

def unpair_gd_npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)
    
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)
    
    npo_loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    npo_loss_val = npo_loss_val.mean()
    
    utility_input_ids = inputs['utility_input_ids'].to(device)
    utility_labels = inputs['utility_labels'].to(device)
    utility_attention_mask = inputs['utility_attention_mask'].to(device)
    utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
    utility_loss = -utility_log_probs.mean()
    
    total_loss = alpha * npo_loss_val + (1 - alpha) * utility_loss
    return total_loss

def pareto_npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)
    
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)
    
    npo_loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    
    utility_input_ids = inputs['utility_input_ids'].to(device)
    utility_labels = inputs['utility_labels'].to(device)
    utility_attention_mask = inputs['utility_attention_mask'].to(device)
    utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_log_probs = -get_sequence_log_probs(utility_outputs.logits, utility_labels)
    
    npo_loss_val = npo_loss_val.mean()
    utility_loss = utility_log_probs.mean()

    return npo_loss_val, utility_loss

def sigmoid_weights(dpo_model_log_probs, ref_log_probs, gamma=1.0):
    rewards = dpo_model_log_probs - ref_log_probs
    weights = 1 - torch.sigmoid(gamma * rewards)
    return weights

def exp_weights(dpo_model_log_probs, ref_log_probs, tau):
    rewards = ref_log_probs - dpo_model_log_probs
    weights = torch.exp(rewards / tau)
    return weights

def wgdnpo_loss(main_model, ref_model, dpo_model, inputs, beta=0.5, retain_loss=True, alpha=0.2,
                adaptive_method='sigmoid', gamma=1.0, tau=5.0):
    device = next(main_model.parameters()).device
    preferred_input_ids = inputs['dpo_preferred_input_ids'].to(device)
    preferred_labels = inputs['dpo_preferred_labels'].to(device)
    preferred_attention_mask = inputs['dpo_preferred_attention_mask'].to(device)
    non_preferred_input_ids = inputs['dpo_non_preferred_input_ids'].to(device)
    non_preferred_labels = inputs['dpo_non_preferred_labels'].to(device)
    non_preferred_attention_mask = inputs['dpo_non_preferred_attention_mask'].to(device)

    preferred_outputs_main = main_model(preferred_input_ids, attention_mask=preferred_attention_mask)
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        preferred_outputs_ref = ref_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    preferred_log_probs_main = get_log_probs(preferred_outputs_main.logits, preferred_labels)
    non_preferred_log_probs_main = get_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    preferred_log_probs_ref = get_log_probs(preferred_outputs_ref.logits, preferred_labels)
    non_preferred_log_probs_ref = get_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    if adaptive_method == 'sigmoid':
        with torch.no_grad():
            dpo_outputs = dpo_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        dpo_log_probs = get_log_probs(dpo_outputs.logits, preferred_labels)
        adaptive_weights = sigmoid_weights(dpo_log_probs, preferred_log_probs_ref, gamma)
    elif adaptive_method == 'exp':
        with torch.no_grad():
            dpo_outputs = dpo_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        dpo_log_probs = get_log_probs(dpo_outputs.logits, preferred_labels)
        adaptive_weights = exp_weights(dpo_log_probs, preferred_log_probs_ref, tau)

    weighted_log_probs = adaptive_weights * preferred_log_probs_main
    gd_loss_val = -weighted_log_probs.sum(dim=-1).mean()

    seq_lengths = torch.sum(non_preferred_attention_mask, dim=1)
    non_preferred_outputs_main_seq = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    non_preferred_log_probs_main_seq = get_sequence_log_probs(non_preferred_outputs_main_seq.logits, non_preferred_labels)
    normalized_log_probs = non_preferred_log_probs_main_seq / seq_lengths
    npo_loss_val = -F.logsigmoid(-beta * normalized_log_probs - gamma)
    npo_loss_val = npo_loss_val.sum(dim=-1).mean()

    wgdnpo_loss_val = gd_loss_val + (npo_loss_val * 2 / beta)
    
    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        final_loss = alpha * wgdnpo_loss_val + (1 - alpha) * utility_loss
    else:
        final_loss = wgdnpo_loss_val

    return final_loss

# Trainer classes
class GATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return ga_loss(model, inputs)

class NPOTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
    def compute_loss(self, model, inputs, return_outputs=False):
        return npo_loss(model, self.rel_model, inputs)

class GDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return gd_loss(model, inputs)

class DPOTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
    def compute_loss(self, model, inputs, return_outputs=False):
        return dpo_loss(model, self.rel_model, inputs)

class DOORTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
    def compute_loss(self, model, inputs, return_outputs=False):
        return gd_npo_loss(model, self.rel_model, inputs)

class WDOORSIGTrainer(Trainer):
    def __init__(self, ref_model, dpo_model, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
        self.dpo_model = dpo_model
        self.gamma = gamma
    def compute_loss(self, model, inputs, return_outputs=False):
        return wgdnpo_loss(model, self.rel_model, self.dpo_model, inputs, gamma=self.gamma, adaptive_method='sigmoid')

class WDOORTrainer(Trainer):
    def __init__(self, ref_model, dpo_model, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
        self.dpo_model = dpo_model
        self.tau = tau
    def compute_loss(self, model, inputs, return_outputs=False):
        return wgdnpo_loss(model, self.rel_model, self.dpo_model, inputs, adaptive_method='exp', tau=self.tau)

class UnpairDOORTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
    def compute_loss(self, model, inputs, return_outputs=False):
        return unpair_gd_npo_loss(model, self.rel_model, inputs)

class ParetoDOORTrainer(Trainer):
    def __init__(self, *args, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_model = ref_model
    def compute_loss(self, model, inputs, return_outputs=False):
        npo_loss_val, utility_loss = pareto_npo_loss(model, self.rel_model, inputs)
        return npo_loss_val, utility_loss


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            npo_loss_val, utility_loss = self.compute_loss(model, inputs)

        loss = npo_loss_val * 0.2 + utility_loss * 0.8
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            npo_loss_val = npo_loss_val.mean()  # mean() to average on multi-gpu parallel training
            utility_loss = utility_loss.mean()  # mean() to average on multi-gpu parallel training

        return loss, npo_loss_val, utility_loss
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (:obj:`nn.Module`):
    #             The model to train.
    #         inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         :obj:`torch.Tensor`: The tensor with training loss on this batch.
    #     """

    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     if self.use_amp:
    #         with autocast():
    #             npo_loss_val, utility_loss = self.compute_loss(model, inputs)
    #     else:
    #         npo_loss_val, utility_loss = self.compute_loss(model, inputs)

    #     loss = npo_loss_val * 0.2 + utility_loss * 0.8

    #     if self.args.n_gpu > 1:
    #         npo_loss_val = npo_loss_val.mean()  # mean() to average on multi-gpu parallel training
    #         utility_loss = utility_loss.mean()  # mean() to average on multi-gpu parallel training
    #         loss = loss.mean()

    #     if self.args.gradient_accumulation_steps > 1:
    #         npo_loss_val = npo_loss_val / self.args.gradient_accumulation_steps
    #         utility_loss = utility_loss / self.args.gradient_accumulation_steps
    #         loss = loss / self.args.gradient_accumulation_steps

    #     if self.use_amp:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         # calling on DS engine (model_wrapped == DDP(Deepspeed(PretrainedModule)))
    #         self.model_wrapped.module.backward(loss)
    #     else:
    #         loss.backward()

    #     return loss.detach(), npo_loss_val, utility_loss

    # def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
    #     """
    #     Main training entry point.

    #     Args:
    #         model_path (:obj:`str`, `optional`):
    #             Local path to the model if the model to train has been instantiated from a local path. If present,
    #             training will resume from the optimizer/scheduler states loaded here.
    #         trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
    #             The trial run or the hyperparameter dictionary for hyperparameter search.
    #     """
    #     # This might change the seed so needs to run first.
    #     self._hp_search_setup(trial)

    #     # Model re-init
    #     if self.model_init is not None:
    #         # Seed must be set before instantiating the model when using model_init.
    #         set_seed(self.args.seed)

    #         model = self.call_model_init(trial)
    #         if not self.is_model_parallel:
    #             model = model.to(self.args.device)

    #         self.model = model
    #         self.model_wrapped = model

    #         # Reinitializes optimizer and scheduler
    #         self.optimizer, self.lr_scheduler = None, None

    #     # Keeping track whether we can can len() on the dataset or not
    #     train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     if train_dataset_is_sized:
    #         num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         if self.args.max_steps > 0:
    #             max_steps = self.args.max_steps
    #             num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
    #                 self.args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #         else:
    #             max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(self.args.num_train_epochs)
    #     else:
    #         # see __init__. max_steps is set when the dataset has no __len__
    #         max_steps = self.args.max_steps
    #         num_train_epochs = 1
    #         num_update_steps_per_epoch = max_steps

    #     if self.args.deepspeed:
    #         model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
    #         self.model = model.module
    #         self.model_wrapped = model  # will get further wrapped in DDP
    #         self.deepspeed = model  # DeepSpeedEngine object
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #     else:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(model_path)

    #     model = self.model_wrapped

    #     # Mixed precision training with apex (torch < 1.6)
    #     if self.use_apex:
    #         model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    #     # Multi-gpu training (should be after apex fp16 initialization)
    #     if self.args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)

    #     # Distributed training (should be after apex fp16 initialization)
    #     if self.sharded_dpp:
    #         model = ShardedDDP(model, self.optimizer)
    #     elif self.args.local_rank != -1:
    #         model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             device_ids=[self.args.local_rank],
    #             output_device=self.args.local_rank,
    #             find_unused_parameters=(
    #                 not getattr(model.config, "gradient_checkpointing", False)
    #                 if isinstance(model, PreTrainedModel)
    #                 else True
    #             ),
    #         )
    #         # find_unused_parameters breaks checkpointing as per
    #         # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

    #     # Train!
    #     if is_torch_tpu_available():
    #         total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
    #     else:
    #         total_train_batch_size = (
    #             self.args.train_batch_size
    #             * self.args.gradient_accumulation_steps
    #             * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
    #         )

    #     num_examples = (
    #         self.num_examples(train_dataloader)
    #         if train_dataset_is_sized
    #         else total_train_batch_size * self.args.max_steps
    #     )

    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples}")
    #     logger.info(f"  Num Epochs = {num_train_epochs}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    #     logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0

    #     # Check if continuing training from a checkpoint
    #     if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
    #         self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not self.args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not self.args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
    #                 "batches in the first epoch."
    #             )

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    #     self.state.trial_params = hp_params(trial) if trial is not None else None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(self.args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = 0
    #     self._total_flos = self.state.total_flos
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not self.args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             # We just need to begin an iteration to create the randomization of the sampler.
    #             for _ in train_dataloader:
    #                 break

    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)

    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
    #                 self.args.device
    #             )
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if self.args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
    #         self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

    #         for step, inputs in enumerate(epoch_iterator):

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 continue

    #             if (step + 1) % self.args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

    #             if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     new_tr_loss, npo_loss_val, utility_loss = self.training_step(model, inputs)
    #                     tr_loss += new_tr_loss
    #             else:
    #                 new_tr_loss, npo_loss_val, utility_loss = self.training_step(model, inputs)
    #                 tr_loss += new_tr_loss
    #             self._total_flos += self.floating_point_ops(inputs)

    #             if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= self.args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping

    #                     if self.use_amp:
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)

    #                     if hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(self.args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         torch.nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
    #                             self.args.max_grad_norm,
    #                         )

    #                 # 这里添加了一步
    #                 losses = [npo_loss_val, utility_loss]
    #                 loss_names = ['npo', 'utility']

    #                 grads = {}

    #                 for idx, loss_type in enumerate(loss_names):
    #                     loss = losses[idx]
    #                     loss.backward(retain_graph=True)

    #                     grads[loss_type] = {}
    #                     for name, param in model.named_parameters():
    #                         if param.grad is not None:
    #                             grads[loss_type][name] = param.grad.detech().clone()

    #                     grads[loss_type]["concat"] = torch.cat(
    #                         [grads[loss_type][name].flatten() for name in grads[loss_type] if name != "concat"]
    #                     )

    #                 # 看看两个梯度更新方向一不一样
    #                 cos_sim = F.cosine_similarity(grads['npo']["concat"], grads['utility']["concat"], dim=0)
    #                 if cos_sim < 0:
    #                     from min_norm_solvers import MinNormSolver
    #                     grads_list = [
    #                         list(grads['npo'].values()),
    #                         list(grads['utility'].values()),
    #                     ]

    #                     weights, min_norm = MinNormSolver.find_min_norm_element(grads_list)
                    
    #                 else:
    #                     weights = [0.5, 0.5]

    #                 gamma = 1.5

    #                 # 更新新的梯度
    #                 for name, param in model.named_parameters():
    #                     three_norm=torch.norm(param.grad.data.clone())
    #                     new_grad=2*weights[0]*grads['npo'][name]+2*weights[1]*grads['utility'][name]
    #                     new_norm=torch.norm(new_grad)
    #                     diff=three_norm/new_norm                        
    #                     if(diff>1):
    #                         param.grad=diff*new_grad*gamma
    #                     else:
    #                         param.grad=new_grad*gamma

    #                 # Optimizer step
    #                 if is_torch_tpu_available():
    #                     xm.optimizer_step(self.optimizer)
    #                 elif self.use_amp:
    #                     self.scaler.step(self.optimizer)
    #                     self.scaler.update()
    #                 else:
    #                     self.optimizer.step()

    #                 self.lr_scheduler.step()
    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break

    #         self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

    #         if self.args.tpu_metrics_debug or self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if self.args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         logger.info(
    #             f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
    #         )
    #         if isinstance(self.model, PreTrainedModel):
    #             self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
    #             if not self.is_model_parallel:
    #                 self.model = self.model.to(self.args.device)
    #         else:
    #             state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
    #             self.model.load_state_dict(state_dict)

    #         if self.deepspeed:
    #             self.deepspeed.load_checkpoint(
    #                 self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
    #             )

    #     metrics = speed_metrics("train", start_time, self.state.max_steps)
    #     if self._total_flos is not None:
    #         self.store_flos()
    #         metrics["total_flos"] = self.state.total_flos
    #     self.log(metrics)

    #     self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()

    #     return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):
            self.accelerator.free_memory()
            self._train_batch_size = batch_size
            if self.args.auto_find_batch_size:
                if self.state.train_batch_size != self._train_batch_size:
                    from accelerate.utils import release_memory

                    (self.model_wrapped,) = release_memory(self.model_wrapped)
                    self.model_wrapped = self.model

                    # Check for DeepSpeed *after* the intial pass and modify the config
                    if self.is_deepspeed_enabled:
                        # Temporarily unset `self.args.train_batch_size`
                        original_bs = self.args.per_device_train_batch_size
                        self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                        self.propagate_args_to_deepspeed(True)
                        self.args.per_device_train_batch_size = original_bs
                self.state.train_batch_size = self._train_batch_size
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()
            if self.is_fsdp_xla_v2_enabled:
                train_dataloader = tpu_spmd_dataloader(train_dataloader)

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            len_dataloader = None
            num_train_tokens = None
            if has_length(train_dataloader):
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_examples = self.num_examples(train_dataloader)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                    if args.include_tokens_per_second:
                        num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                        )
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                    if args.include_tokens_per_second:
                        num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
            elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_examples = total_train_batch_size * args.max_steps
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {args.max_steps}"
                )

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                        " (torchrun or torch.distributed.launch (deprecated))."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

            # We need to reset the scheduler, as its parameters may be different on subsequent calls
            if self._created_lr_scheduler:
                self.lr_scheduler = None
                self._created_lr_scheduler = False

            if self.is_deepspeed_enabled:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState(
                stateful_callbacks=[
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]
            )
            self.state.is_hyper_param_search = trial is not None
            self.state.train_batch_size = self._train_batch_size

            # Compute absolute values for logging, eval, and save if given as ratio
            if args.logging_steps is not None:
                if args.logging_steps < 1:
                    self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
                else:
                    self.state.logging_steps = args.logging_steps
            if args.eval_steps is not None:
                if args.eval_steps < 1:
                    self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
                else:
                    self.state.eval_steps = args.eval_steps
            if args.save_steps is not None:
                if args.save_steps < 1:
                    self.state.save_steps = math.ceil(max_steps * args.save_steps)
                else:
                    self.state.save_steps = args.save_steps

            # Activate gradient checkpointing if needed
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

            model = self._wrap_model(self.model_wrapped)

            # as the model is wrapped, don't use `accelerator.prepare`
            # this is for unhandled cases such as
            # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                if use_accelerator_prepare:
                    self._fsdp_qlora_plugin_updates()
                    self.model = self.accelerator.prepare(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # prepare using `accelerator` prepare
            if use_accelerator_prepare:
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )
            elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                # In this case we are in DDP + LOMO, which should be supported
                self.optimizer = self.accelerator.prepare(self.optimizer)

            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            # ckpt loading
            if resume_from_checkpoint is not None:
                if self.is_deepspeed_enabled:
                    deepspeed_load_checkpoint(
                        self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                    )
                elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
            # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                self.compare_trainer_and_checkpoint_args(self.args, self.state)
                self._load_callback_state()
                epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            if self.hp_name is not None and self._trial is not None:
                # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
                # parameter to Train when using DDP.
                self.state.trial_name = self.hp_name(self._trial)
            if trial is not None:
                assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
                self.state.trial_params = hp_params(assignments)
            else:
                self.state.trial_params = None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()
            grad_norm: Optional[float] = None
            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            if args.eval_on_start:
                self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

            total_batched_samples = 0
            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                for step, inputs in enumerate(epoch_iterator):
                    total_batched_samples += 1

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += (
                                torch.sum(
                                    self.accelerator.gather(
                                        torch.tensor(
                                            inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                        )
                                    )
                                )
                                .cpu()
                                .item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        # TODO
                        all_loss, npo_loss_val, utility_loss = self.training_step(model, inputs)
                        tr_loss_step = all_loss.detach() / self.args.gradient_accumulation_steps
                        # tr_loss_step = all_loss.detach()

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        # 这里添加了一步
                        losses = [npo_loss_val, utility_loss]
                        loss_names = ['npo', 'utility']

                        grads = {}

                        for idx, loss_type in enumerate(loss_names):
                            loss = losses[idx]
                            loss.backward(retain_graph=True)

                            grads[loss_type] = {}

                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    grads[loss_type][name] = param.grad.data.clone()

                            grads[loss_type]["concat"] = torch.cat(
                                [grads[loss_type][name].flatten() for name in grads[loss_type] if name != "concat"]
                            )

                        # 看看两个梯度更新方向一不一样
                        cos_sim = F.cosine_similarity(grads['npo']["concat"], grads['utility']["concat"], dim=0)
                        if cos_sim < 0:
                            from min_norm_solvers import MinNormSolver
                            grads_list = [
                                list(grads['npo'].values()),
                                list(grads['utility'].values()),
                            ]

                            weights, min_norm = MinNormSolver.find_min_norm_element(grads_list)
                        
                        else:
                            weights = [0.2, 0.8]

                        gamma = 1

                        if self.use_apex:
                            with amp.scale_loss(all_loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            self.accelerator.backward(all_loss)

                        # 更新新的梯度
                        for name, param in model.named_parameters():
                            if param.grad is None:
                                continue
                            # three_norm=torch.norm(param.grad.data.clone())
                            new_grad=2*weights[0]*grads['npo'][name]+2*weights[1]*grads['utility'][name]
                            # new_norm=torch.norm(new_grad)
                            # diff=three_norm/new_norm                        
                            # if diff > 1:
                            #     scale_factor = max(diff, 10) * gamma
                            #     param.grad.copy_(new_grad * scale_factor)
                            # else:
                            param.grad.copy_(new_grad * gamma)
                            # if diff > 1:
                            #     scale_factor = max(diff, 10) * gamma
                            #     param.grad = new_grad * scale_factor
                            # else:
                            #     param.grad = new_grad * gamma

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                if step < 0:
                    logger.warning(
                        "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_xla_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sure the model has been saved by process 0.
                if is_torch_xla_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                    dist.barrier()
                elif is_sagemaker_mp_enabled():
                    smp.barrier()

                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
            train_loss = self._total_loss_scalar / effective_global_step

            metrics = speed_metrics(
                "train",
                start_time,
                num_samples=num_train_samples,
                num_steps=self.state.max_steps,
                num_tokens=num_train_tokens,
            )
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint, ignore_errors=True)

            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            # Wait for the checkpoint to be uploaded.
            self._finish_current_push()

            # After training we make sure to retrieve back the original forward pass method
            # for the embedding layer by removing the forward post hook.
            if self.neftune_noise_alpha is not None:
                self._deactivate_neftune(self.model)

            return TrainOutput(self.state.global_step, train_loss, metrics)