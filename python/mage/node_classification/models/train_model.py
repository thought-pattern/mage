"""Utilities for train model."""

from importlib import import_module

from torch import Tensor as torch_Tensor


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
        torch.tensor: loss calculated when training step is performed
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
            float: returns loss calculated during training or validation
        """
        ret = 0.0
        batch_count = 0

        # Set the model to train or eval mode depending on the flag gradient.
        if gradient:
            model.train()
        else:
            model.eval()

        for batch in loader:
            batch_count += 1
            if gradient:
                opt.zero_grad()  # Clear gradients.

            model_output = model(batch.x_dict, batch.edge_index_dict)
            if not isinstance(model_output, dict):
                raise TypeError(f"Node-classification model returned {type(model_output)}, expected dict")
            out = model_output.get(observed_attribute, False)
            if not isinstance(out, torch_Tensor):
                raise KeyError(f"Model output does not contain tensor data for {observed_attribute}")
            loss = criterion(
                out, batch[observed_attribute].y
            )  # Compute the loss solely based on the training nodes.
            if not isinstance(loss, torch_Tensor):
                raise TypeError(f"Training criterion returned {type(loss)}, expected torch.Tensor")

            if gradient:
                loss.backward()  # Derive gradients.
                opt.step()  # Update parameters based on gradients.

            ret += loss.item()

        computed_return_value = ret / batch_count if batch_count else 0.0
        return computed_return_value

    ret = training_loop(train_loader, True)
    ret_val = training_loop(val_loader, False)

    return ret, ret_val
