"""Utilities for train model."""

from importlib import import_module

from torch import (
    Tensor as torch_Tensor,
    arange as torch_arange,
    bool as torch_bool,
    cat as torch_cat,
    ones as torch_ones,
    randperm as torch_randperm,
    set_grad_enabled as torch_set_grad_enabled,
    zeros_like as torch_zeros_like,
)
from torch_geometric.data import HeteroData


def build_seed_first_batches(data: HeteroData, observed_attribute: str, mask_tensor: torch_Tensor, batch_size: int, shuffle: bool) -> list:
    """Fallback pure-PyTorch HeteroData mini-batcher placing each batch's seed nodes at rows 0..seed_count-1."""
    seed_indices = mask_tensor.nonzero(as_tuple=False).view(-1)
    total_seeds = int(seed_indices.numel())
    if total_seeds == 0:
        return []
    if shuffle and total_seeds > 1:
        order = torch_randperm(total_seeds, device=seed_indices.device)
        seed_indices = seed_indices[order]

    num_obs = int(data[observed_attribute].x.size(0))
    all_obs = torch_arange(num_obs, device=seed_indices.device)
    batches = []

    for offset in range(0, total_seeds, batch_size):
        chunk = seed_indices[offset : offset + batch_size]
        seed_count = int(chunk.numel())
        is_seed = torch_ones((num_obs,), dtype=torch_bool, device=seed_indices.device)
        is_seed[chunk] = False
        rest = all_obs[is_seed]
        perm = torch_cat([chunk, rest], dim=0)
        inv_perm = torch_zeros_like(perm)
        inv_perm[perm] = all_obs

        batch = HeteroData()
        for ntype in data.node_types:
            if ntype == observed_attribute:
                batch[ntype].x = data[ntype].x[perm]
                batch[ntype].y = data[ntype].y[perm]
                batch[ntype].batch_size = seed_count
            else:
                batch[ntype].x = data[ntype].x
                if hasattr(data[ntype], "y"):
                    batch[ntype].y = data[ntype].y

        for etype in data.edge_types:
            src_type, _, dst_type = etype
            ei = data[etype].edge_index
            src_idx = inv_perm[ei[0]] if src_type == observed_attribute else ei[0]
            dst_idx = inv_perm[ei[1]] if dst_type == observed_attribute else ei[1]
            batch[etype].edge_index = ei.new_stack([src_idx, dst_idx]) if hasattr(ei, "new_stack") else torch_cat([src_idx.unsqueeze(0), dst_idx.unsqueeze(0)], dim=0)

        batches.append(batch)

    return batches


def train_epoch(
    model,
    opt,
    data,
    criterion,
    batch_size: int,
    observed_attribute: str,
    num_samples: dict,
) -> tuple[float, float]:
    """Perform one epoch of training and validation, restricting loss to batch seed nodes."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, received {batch_size}")
    if not observed_attribute:
        raise ValueError("observed_attribute must not be empty")
    if not isinstance(num_samples, dict) or not num_samples:
        raise ValueError("num_samples must be a non-empty dict")
    try:
        loader_module = import_module("torch_geometric.loader")
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("Node-classification training requires torch-geometric") from err
    loader_type = getattr(loader_module, "HGTLoader", False)
    if not callable(loader_type):
        raise ImportError("torch_geometric.loader does not provide HGTLoader")

    observed_data = data[observed_attribute]
    has_train = bool(observed_data.train_mask.any().item())
    has_val = bool(observed_data.val_mask.any().item())

    def training_loop(mask_tensor, shuffle: bool, gradient: bool) -> float:
        """Run one pass over seed nodes selected by mask_tensor."""
        if gradient:
            model.train()
        else:
            model.eval()

        if not bool(mask_tensor.any().item()):
            return 0.0

        try:
            loader = loader_type(
                data=data,
                num_samples=num_samples,
                shuffle=shuffle,
                batch_size=batch_size,
                input_nodes=(observed_attribute, mask_tensor),
            )
        except ImportError:
            loader = build_seed_first_batches(data, observed_attribute, mask_tensor, batch_size, shuffle)

        loss_sum = 0.0
        seed_total = 0

        for batch in loader:
            observed_batch = batch[observed_attribute]
            seed_count = int(getattr(observed_batch, "batch_size", 0))
            if seed_count < 1:
                continue

            if gradient:
                opt.zero_grad()

            with torch_set_grad_enabled(gradient):
                model_output = model(batch.x_dict, batch.edge_index_dict)
                if not isinstance(model_output, dict):
                    raise TypeError(f"Node-classification model returned {type(model_output)}, expected dict")
                out = model_output.get(observed_attribute, False)
                if not isinstance(out, torch_Tensor):
                    raise KeyError(f"Model output does not contain tensor data for {observed_attribute}")
                loss = criterion(out[:seed_count], observed_batch.y[:seed_count])
                if not isinstance(loss, torch_Tensor):
                    raise TypeError(f"Training criterion returned {type(loss)}, expected torch.Tensor")

            if gradient:
                loss.backward()
                opt.step()

            loss_sum += loss.item() * seed_count
            seed_total += seed_count

        computed_return_value = loss_sum / seed_total if seed_total else 0.0
        return computed_return_value

    ret = training_loop(observed_data.train_mask, True, True) if has_train else 0.0
    ret_val = training_loop(observed_data.val_mask, False, False) if has_val else 0.0

    return ret, ret_val
