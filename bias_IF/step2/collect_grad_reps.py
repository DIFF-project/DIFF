import json
import os
# import umap
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel
from sklearn.manifold import TSNE
import numpy as np
from typing import Optional, List
import pdb
from peft import PeftModel


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
        # pdb.set_trace()
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    # loss = model(**batch).loss
    outputs = model(**batch)
    loss = outputs.loss if hasattr(outputs, 'loss') else None
    label = batch['input_ids'][:, 1:]

    # FIXME 不知道对不对
    if loss is None:
        logits = outputs.logits
        logits = logits[:, :-1, :]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    # loss = model(**batch).loss
    loss = outputs.loss if hasattr(outputs, 'loss') else None

    # FIXME 不知道对不对
    if loss is None:
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), batch['input_ids'].view(-1))
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    # loss = model(**batch).loss
    outputs = model(**batch)
    loss = outputs.loss if hasattr(outputs, 'loss') else None
    label = batch['input_ids'][:, 1:]

    # FIXME 不知道对不对
    if loss is None:
        logits = outputs.logits
        logits = logits[:, :-1, :]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), label.reshape(-1))

    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads

def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    # for key in optimizer_state.keys():
    #     print(f"Key: {key}")
    # avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    # pdb.set_trace()
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in range(len(names))])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in range(len(names))])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

# def prepare_optimizer_state(model, optimizer_state, device):
#     names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
#     # for key in optimizer_state.keys():
#     #     print(f"Key: {key}")
#     # avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
#     if len(names) != len(optimizer_state):
#         print(f"Warning: Number of trainable parameters ({len(trainable_params)}) "
#             f"doesn't match optimizer states ({len(adam_optimizer_state)})")
#         print("\nTrainable parameter names:")
#         pdb.set_trace()
#     avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
#     avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in names])
#     avg = avg.to(device)
#     avg_sq = avg_sq.to(device)
#     return avg, avg_sq

# def prepare_optimizer_state(model, optimizer_state, device):
#     names_and_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
#     min_key = min(int(k) for k in optimizer_state.keys())
    
#     state_tensors = []
#     state_sq_tensors = []
    
#     for i, (name, param) in enumerate(names_and_params):
#         state_key = min_key + i
#         if state_key in optimizer_state:
#             exp_avg = optimizer_state[state_key]["exp_avg"].to(device).view(-1)
#             exp_avg_sq = optimizer_state[state_key]["exp_avg_sq"].to(device).view(-1)
#             state_tensors.append(exp_avg)
#             state_sq_tensors.append(exp_avg_sq)
#             print(f"Mapped {name} to optimizer state key {state_key}")
    
#     if not state_tensors:
#         raise ValueError("No optimizer states found for any parameters")
    
#     avg = torch.cat(state_tensors)
#     avg_sq = torch.cat(state_sq_tensors)
    
#     return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):

    model_id = 0 
    block_size = 2
    projector_batch_size = 64
    torch.random.manual_seed(0) 

    project_interval = 1
    save_interval = 160

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.normal,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    full_grads = [] 
    projected_grads = {dim: [] for dim in proj_dim}  


    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)
        count += 1

        if count <= max_index:
            print("skipping count", count)
            continue

        # TODO 梯度是一个值，对每个batch取平均，最后跑实验可以试试不同给的batch
        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)

        full_grads.append(vectorized_grads)
        model.zero_grad()

        if count % project_interval == 0:
            # pdb.set_trace()
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    
    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))


# def map_projection(grads: torch.Tensor, n_components: int):
#     # 计算协方差矩阵
#     cov = torch.mm(grads.t(), grads) / grads.shape[0]
    
#     # 计算特征值和特征向量
#     eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
#     # 选择最大的n_components个特征值对应的特征向量
#     idx = torch.argsort(eigenvalues, descending=True)[:n_components]
#     components = eigenvectors[:, idx]
    
#     # 投影数据
#     projected = torch.mm(grads, components)
    
#     return projected, components

# def collect_grads(dataloader,
#                   model,
#                   output_dir,
#                   n_components: int = 8192,
#                   adam_optimizer_state: Optional[dict] = None,
#                   gradient_type: str = "adam",
#                   max_samples: Optional[int] = None,
#                   batch_size = 32):

#     device = next(model.parameters()).device
#     dtype = next(model.parameters()).dtype

#     if gradient_type == "adam":
#         assert adam_optimizer_state is not None
#         m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

#     count = 0
#     os.makedirs(output_dir, exist_ok=True)

#     all_grads = []
#     all_projected_grads = []
#     all_projection_components = []

#     for batch in tqdm(dataloader, total=len(dataloader)):
#         prepare_batch(batch)
#         count += 1

#         if gradient_type == "adam":
#             vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
#         elif gradient_type == "sign":
#             vectorized_grads = obtain_sign_gradients(model, batch)
#         else:
#             vectorized_grads = obtain_gradients(model, batch)
            
#         all_grads.append(vectorized_grads.cpu())
#         model.zero_grad()

#         if len(all_grads) >= batch_size:
#             batch_grads = torch.cat(all_grads, dim=0)
#             pdb.set_trace()
#             projected_grads, projection_components = map_projection(batch_grads, n_components)
            
#             all_projected_grads.append(projected_grads)
#             all_projection_components.append(projection_components)
            
#             all_grads = []

#     # 最后合并所有结果
#     if all_grads:
#         batch_grads = torch.cat(all_grads, dim=0)
#         projected_grads, projection_components = map_projection(batch_grads, n_components)
        
#         all_projected_grads.append(projected_grads)
#         all_projection_components.append(projection_components)

#     # 合并所有投影结果
#     final_projected_grads = torch.cat(all_projected_grads, dim=0)
#     final_projection_components = torch.cat(all_projection_components, dim=1)

#     outfile = os.path.join(output_dir, "map_grads.pt")
#     torch.save({
#         'projected_grads': final_projected_grads,
#         'projection_components': final_projection_components
#     }, outfile)
