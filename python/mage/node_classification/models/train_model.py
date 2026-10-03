"""Utilities for train model."""

from importlib import import_module

from torch import Tensor as torch_Tensor
from torch import set_grad_enabled as torch_set_grad_enabled


def train_epoch(
    model,
    opt,
    data,
    criterion,
    batch_size: int,
    observed_attribute: str,
    num_samples: dict,
) -> tuple[float, float]:
    """In this function, one epoch of training is performed.

    Args:
        model (Any): object for model
        opt (Any): model optimizer
        data (Data): prepared dataset for training
        criterion (Any): criterion for loss calculation
        batch_size (int): batch size for training
        observed_attribute (str): observed attribute for training
        num_samples (dict): The number of nodes to
            sample in each iteration and for each node type.

    Returns:
        tuple[float, float]: training and validation loss, each the mean over the epoch's seed nodes
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, received {batch_size}")
    if not observed_attribute:
        raise ValueError("observed_attribute must not be empty")
    if not isinstance(num_samples, dict) or not num_samples:
        raise ValueError("num_samples must be a non-empty dict")
    try:
        loader_module = import_module("torch_geometric.loader")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Node-classification training requires torch-geometric") from error
    loader_type = getattr(loader_module, "HGTLoader", False)
    if not callable(loader_type):
        raise ImportError("torch_geometric.loader does not provide HGTLoader")

    observed_data = data[observed_attribute]
    train_input_nodes = (observed_attribute, observed_data.train_mask)
    val_input_nodes = (observed_attribute, observed_data.val_mask)

    train_loader = loader_type(
        data=data,
        num_samples=num_samples,
        shuffle=True,
        batch_size=batch_size,
        input_nodes=train_input_nodes,
    )

    val_loader = loader_type(
        data=data,
        num_samples=num_samples,
        shuffle=False,
        batch_size=batch_size,
        input_nodes=val_input_nodes,
    )

    def training_loop(loader, gradient: bool) -> float:
        """Loop for either train or validation, depending on the flag gradient.

        Args:
            loader (HGTLoader): train or validation loader
            gradient (bool): True for train, False for validation

        Returns:
            float: mean loss per seed node over the loader's batches
        """
        loss_sum = 0.0
        seed_total = 0

        # Set the model to train or eval mode depending on the flag gradient.
        if gradient:
            model.train()
        else:
            model.eval()

        for batch in loader:
            # The loader places this batch's seed nodes of the observed type first. The rows after them are sampled
            # neighbours that only give message-passing context, and their labels may belong to the other split, so
            # labels and loss are restricted to the seed prefix.
            observed_batch = batch[observed_attribute]
            seed_count = int(getattr(observed_batch, "batch_size", 0))
            if seed_count < 1:
                raise ValueError(f"Loader batch for {observed_attribute} carries no seed nodes")

            if gradient:
                opt.zero_grad()  # Clear gradients.

            # Validation must not record an autograd graph; training needs one for the backward pass.
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
                loss.backward()  # Derive gradients.
                opt.step()  # Update parameters based on gradients.

            # The criterion averages over its seed rows; weighting by the seed count makes the epoch value the mean
            # over seed nodes even when the last batch is smaller.
            loss_sum += loss.item() * seed_count
            seed_total += seed_count

        computed_return_value = loss_sum / seed_total if seed_total else 0.0
        return computed_return_value

    ret = training_loop(train_loader, True)
    ret_val = training_loop(val_loader, False)

    return ret, ret_val
