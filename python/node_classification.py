"""Utilities for node classification."""

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite
from os import (
    fdopen as os_fdopen,
    fsync as os_fsync,
    getcwd as os_getcwd,
    listdir as os_listdir,
    makedirs as os_makedirs,
    path as os_path,
    remove as os_remove,
    replace as os_replace,
)
from re import compile as re_compile
from tempfile import gettempdir, mkstemp
from time import time

from mgp import (
    Any as mgp_Any,
    List as mgp_List,
    Nullable as mgp_Nullable,
    ProcCtx as mgp_ProcCtx,
    Record as mgp_Record,
    Vertex as mgp_Vertex,
    read_proc as mgp_read_proc,
)
from torch import (
    cuda as torch_cuda,
    load as torch_load,
    nn as torch_nn,
    no_grad as torch_no_grad,
    optim as torch_optim,
    save as torch_save,
    zeros as torch_zeros,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from tqdm import tqdm

from mage.node_classification.models.gat import GAT
from mage.node_classification.models.gatjk import GATJK
from mage.node_classification.models.gatv2 import GATv2
from mage.node_classification.models.sage import SAGE
from mage.node_classification.models.train_model import train_epoch
from mage.node_classification.utils.extract_from_database import extract_from_database
from mage.node_classification.utils.metrics import metrics

DEFAULT_ARGUMENT_DICT = {}


class ModelParams:
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    HIDDEN_FEATURES_SIZE = "hidden_features_size"
    LAYER_TYPE = "layer_type"
    AGGREGATOR = "aggregator"


class OptimizerParams:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


class DataParams:
    SPLIT_RATIO = "split_ratio"
    METRICS = "metrics"


class MemgraphParams:
    NODE_ID_PROPERTY = "node_id_property"


class TrainParams:
    NUM_EPOCHS = "num_epochs"
    CONSOLE_LOG_FREQ = "console_log_freq"
    CHECKPOINT_FREQ = "checkpoint_freq"
    BATCH_SIZE = "batch_size"
    MAX_MODELS_TO_KEEP = "max_models_to_keep"
    TIME_BETWEEN_CHECKPOINTS = "time_between_checkpoints"


class HeteroParams:
    FEATURES_NAME = "features_name"
    OBSERVED_ATTRIBUTE = "observed_attribute"
    CLASS_NAME = "class_name"
    REINDEXING = "reindexing"
    INV_REINDEXING = "inv_reindexing"
    NUM_NODES_SAMPLE = "num_nodes_sample"
    NUM_ITERATIONS_SAMPLE = "num_iterations_sample"
    LABEL_REINDEXING = "label_reindexing"
    INV_LABEL_REINDEXING = "inv_label_reindexing"
    NODE_TYPES = "node_types"
    EDGE_TYPES = "edge_types"


class OtherParams:
    DEVICE_TYPE = "device_type"
    PATH_TO_MODEL = "path_to_model"
    PATIENCE = "patience"
    MODEL_SAVING_FOLDER = "model_saving_folder"


GAT_MODEL = "GAT"
GATV2_MODEL = "GATv2"
SAGE_MODEL = "SAGE"
GAT_WITH_JK = "GATJK"

MODELS = {GAT_MODEL: GAT, GATV2_MODEL: GATv2, SAGE_MODEL: SAGE, GAT_WITH_JK: GATJK}

DEFINED_INPUT_TYPES = {
    ModelParams.HIDDEN_FEATURES_SIZE: list,
    ModelParams.LAYER_TYPE: str,
    TrainParams.NUM_EPOCHS: int,
    OptimizerParams.LEARNING_RATE: float,
    OptimizerParams.WEIGHT_DECAY: float,
    DataParams.SPLIT_RATIO: float,
    MemgraphParams.NODE_ID_PROPERTY: str,
    OtherParams.DEVICE_TYPE: str,
    TrainParams.CONSOLE_LOG_FREQ: int,
    TrainParams.CHECKPOINT_FREQ: int,
    TrainParams.BATCH_SIZE: int,
    TrainParams.MAX_MODELS_TO_KEEP: int,
    TrainParams.TIME_BETWEEN_CHECKPOINTS: float,
    ModelParams.AGGREGATOR: str,
    DataParams.METRICS: list,
    HeteroParams.OBSERVED_ATTRIBUTE: str,
    HeteroParams.FEATURES_NAME: str,
    HeteroParams.CLASS_NAME: str,
    HeteroParams.REINDEXING: dict,
    HeteroParams.INV_REINDEXING: dict,
    HeteroParams.NUM_NODES_SAMPLE: int,
    HeteroParams.NUM_ITERATIONS_SAMPLE: int,
    OtherParams.PATH_TO_MODEL: str,
    OtherParams.PATIENCE: int,
    OtherParams.MODEL_SAVING_FOLDER: str,
}

DEFAULT_VALUES = {
    ModelParams.HIDDEN_FEATURES_SIZE: [16, 16],
    ModelParams.LAYER_TYPE: "GATJK",
    TrainParams.NUM_EPOCHS: 100,
    OptimizerParams.LEARNING_RATE: 0.1,
    OptimizerParams.WEIGHT_DECAY: 5e-4,
    DataParams.SPLIT_RATIO: 0.8,
    MemgraphParams.NODE_ID_PROPERTY: "id",
    OtherParams.DEVICE_TYPE: "cpu",
    TrainParams.CONSOLE_LOG_FREQ: 5,
    TrainParams.CHECKPOINT_FREQ: 5,
    TrainParams.BATCH_SIZE: 64,
    TrainParams.MAX_MODELS_TO_KEEP: 5,
    TrainParams.TIME_BETWEEN_CHECKPOINTS: 2.0,
    ModelParams.AGGREGATOR: "mean",
    DataParams.METRICS: [
        "loss",
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "num_wrong_examples",
    ],
    HeteroParams.OBSERVED_ATTRIBUTE: "",
    HeteroParams.FEATURES_NAME: "features",
    HeteroParams.CLASS_NAME: "class",
    HeteroParams.REINDEXING: {},
    HeteroParams.INV_REINDEXING: {},
    HeteroParams.NUM_NODES_SAMPLE: 512,
    HeteroParams.NUM_ITERATIONS_SAMPLE: 4,
    OtherParams.PATH_TO_MODEL: "",
    OtherParams.PATIENCE: 10,
    OtherParams.MODEL_SAVING_FOLDER: os_path.join(gettempdir(), "torch_models"),
}

model: mgp_Any = False
current_values: dict = deepcopy(DEFAULT_VALUES)
logged_data: mgp_List = []
checkpoint_sequence: int = 0

POSITIVE_INTEGER_PARAMETERS = (
    TrainParams.NUM_EPOCHS,
    TrainParams.CONSOLE_LOG_FREQ,
    TrainParams.CHECKPOINT_FREQ,
    TrainParams.BATCH_SIZE,
    TrainParams.MAX_MODELS_TO_KEEP,
    HeteroParams.NUM_NODES_SAMPLE,
    HeteroParams.NUM_ITERATIONS_SAMPLE,
    OtherParams.PATIENCE,
)
POSITIVE_FLOAT_PARAMETERS = (OptimizerParams.LEARNING_RATE,)
NON_NEGATIVE_FLOAT_PARAMETERS = (
    OptimizerParams.WEIGHT_DECAY,
    TrainParams.TIME_BETWEEN_CHECKPOINTS,
)

CHECKPOINT_TASK_FIELDS = {
    ModelParams.LAYER_TYPE: "",
    ModelParams.IN_CHANNELS: 0,
    ModelParams.HIDDEN_FEATURES_SIZE: [],
    ModelParams.OUT_CHANNELS: 0,
    ModelParams.AGGREGATOR: "",
    HeteroParams.NODE_TYPES: [],
    HeteroParams.EDGE_TYPES: [],
    HeteroParams.OBSERVED_ATTRIBUTE: "",
    HeteroParams.LABEL_REINDEXING: {},
    HeteroParams.INV_LABEL_REINDEXING: {},
    HeteroParams.FEATURES_NAME: "",
    HeteroParams.CLASS_NAME: "",
}
CHECKPOINT_STATE_DICT = "state_dict"
CHECKPOINT_TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"
CHECKPOINT_NAME = re_compile(r"model_(?P<layer_type>[A-Za-z0-9]+)_(?P<saved_at>\d{8}T\d{12}Z)(?:_(?P<seq>\d+))?\.pt")
LEGACY_CHECKPOINT_NAME = re_compile(
    r"model_(?P<layer_type>[A-Za-z0-9]+)_(?P<saved_at>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_(?P<seq>\d+))?\.pt"
)


def resolve_device(requested_device: str) -> str:
    """Resolve the requested device without overriding an explicit CPU selection."""
    normalized = str(requested_device or "cpu").strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch_cuda.is_available():
            raise ValueError("CUDA device requested, while CUDA is not available.")
        return "cuda:0" if normalized == "cuda" else normalized
    raise ValueError(f"Unsupported device_type {requested_device!r}; expected 'cpu' or 'cuda'.")


def declare_data_into(ctx: mgp_ProcCtx, values: dict) -> HeteroData:
    """Extract graph data and record dataset metadata into values."""
    device = resolve_device(values.get(OtherParams.DEVICE_TYPE, "cpu"))
    values[OtherParams.DEVICE_TYPE] = device

    nodes = list(iter(ctx.graph.vertices))
    if not nodes:
        raise ValueError("Graph is empty.")

    (
        data,
        values[HeteroParams.OBSERVED_ATTRIBUTE],
        values[HeteroParams.REINDEXING],
        values[HeteroParams.INV_REINDEXING],
        values[HeteroParams.LABEL_REINDEXING],
        values[HeteroParams.INV_LABEL_REINDEXING],
    ) = extract_from_database(
        nodes,
        values.get(DataParams.SPLIT_RATIO, 0.8),
        values.get(HeteroParams.FEATURES_NAME, "features"),
        values.get(HeteroParams.CLASS_NAME, "class"),
        device,
    )

    observed_attribute = values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")
    if observed_attribute not in data.node_types:
        raise ValueError(f"No node with property '{values.get(HeteroParams.CLASS_NAME, '')}' has features.")
    observed_attribute_data = data[observed_attribute]

    values[ModelParams.IN_CHANNELS] = observed_attribute_data.x.size(dim=1)
    values[ModelParams.OUT_CHANNELS] = len(values.get(HeteroParams.LABEL_REINDEXING, {}))
    if values[ModelParams.OUT_CHANNELS] < 1:
        raise ValueError("No labeled target nodes found for training.")
    values[HeteroParams.NODE_TYPES] = list(data.node_types)
    values[HeteroParams.EDGE_TYPES] = list(data.edge_types)

    return data


def declare_data(ctx: mgp_ProcCtx) -> HeteroData:
    """Initialize dataset metadata in current_values."""
    return declare_data_into(ctx, current_values)


def hetero_network_from_values(values: dict):
    """Build the heterogeneous network described by values."""
    args_gatjk = [
        values.get(ModelParams.IN_CHANNELS, 0),
        values.get(ModelParams.HIDDEN_FEATURES_SIZE, []),
        values.get(ModelParams.OUT_CHANNELS, 0),
    ]
    args_inductive = [
        values.get(ModelParams.IN_CHANNELS, 0),
        values.get(ModelParams.HIDDEN_FEATURES_SIZE, []),
        values.get(ModelParams.OUT_CHANNELS, 0),
        values.get(ModelParams.AGGREGATOR, ""),
    ]

    layer_type = values.get(ModelParams.LAYER_TYPE, "")
    if layer_type not in MODELS:
        raise ValueError(
            "You didn't choose one of currently available models (GAT, GATv2, GATJK and SAGE). Please choose one of them."
        )

    args = args_gatjk if layer_type == GAT_WITH_JK else args_inductive
    network = MODELS.get(layer_type, GAT)(*args)
    metadata = (
        values.get(HeteroParams.NODE_TYPES, []),
        values.get(HeteroParams.EDGE_TYPES, []),
    )
    network = to_hetero(network, metadata)
    network.to(resolve_device(values.get(OtherParams.DEVICE_TYPE, "cpu")))
    return network


def declare_saving_paths(values: dict) -> str:
    """Ensure the model saving folder exists and record the checkpoint prefix in values."""
    path = os_path.join(os_getcwd(), values.get(OtherParams.MODEL_SAVING_FOLDER, DEFAULT_VALUES[OtherParams.MODEL_SAVING_FOLDER]))
    os_makedirs(path, exist_ok=True)
    values[OtherParams.PATH_TO_MODEL] = os_path.join(path, "model_" + values.get(ModelParams.LAYER_TYPE, "GATJK") + "_")
    return path


def configuration_errors(values: dict) -> list[str]:
    """Return every domain error in values."""
    errors = []
    for name in POSITIVE_INTEGER_PARAMETERS:
        value = values.get(name, 0)
        if isinstance(value, bool) or value < 1:
            errors.append(f"{name} must be a positive integer")
    for name in POSITIVE_FLOAT_PARAMETERS:
        value = values.get(name, 0.0)
        if not isfinite(value) or value <= 0:
            errors.append(f"{name} must be a finite positive number")
    for name in NON_NEGATIVE_FLOAT_PARAMETERS:
        value = values.get(name, 0.0)
        if not isfinite(value) or value < 0:
            errors.append(f"{name} must be a finite non-negative number")
    split_ratio = values.get(DataParams.SPLIT_RATIO, 0.0)
    if not isfinite(split_ratio) or not 0 < split_ratio < 1:
        errors.append(f"{DataParams.SPLIT_RATIO} must be a finite number strictly between 0 and 1")
    hidden_features_size = values.get(ModelParams.HIDDEN_FEATURES_SIZE, [])
    if not hidden_features_size or any(
        not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in hidden_features_size
    ):
        errors.append(f"{ModelParams.HIDDEN_FEATURES_SIZE} must be a non-empty list of positive integers")
    device_val = str(values.get(OtherParams.DEVICE_TYPE, "")).strip().lower()
    if device_val != "cpu" and device_val != "cuda" and not device_val.startswith("cuda:"):
        errors.append(f"{OtherParams.DEVICE_TYPE} must be 'cpu' or 'cuda'")
    return errors


def set_model_parameters(
    params: mgp_Any = DEFAULT_ARGUMENT_DICT,
) -> mgp_Record:
    """Validate and store model/training parameters."""
    if params is DEFAULT_ARGUMENT_DICT:
        params = DEFAULT_ARGUMENT_DICT.copy()
    global current_values

    candidate = deepcopy(DEFAULT_VALUES)
    candidate.update(params)

    for list_parameter in (ModelParams.HIDDEN_FEATURES_SIZE, DataParams.METRICS):
        if isinstance(candidate.get(list_parameter, []), tuple):
            candidate[list_parameter] = list(candidate.get(list_parameter, []))

    correctly_typed = all(
        name in candidate
        and isinstance(candidate.get(name, expected_type()), expected_type)
        and not (expected_type is int and isinstance(candidate.get(name), bool))
        and not (expected_type is float and isinstance(candidate.get(name), bool))
        for name, expected_type in DEFINED_INPUT_TYPES.items()
    )
    if not correctly_typed:
        raise TypeError("Input dictionary is not correctly typed.")
    errors = configuration_errors(candidate)
    if errors:
        raise ValueError("Input dictionary is out of range: " + "; ".join(errors))

    declare_saving_paths(candidate)
    current_values = candidate

    computed_return_value = mgp_Record(
        hidden_features_size=current_values.get(ModelParams.HIDDEN_FEATURES_SIZE, []),
        layer_type=current_values.get(ModelParams.LAYER_TYPE, ""),
        aggregator=current_values.get(ModelParams.AGGREGATOR, ""),
        learning_rate=current_values.get(OptimizerParams.LEARNING_RATE, 0.0),
        weight_decay=current_values.get(OptimizerParams.WEIGHT_DECAY, 0.0),
        split_ratio=current_values.get(DataParams.SPLIT_RATIO, 0.0),
        metrics=current_values.get(DataParams.METRICS, []),
        node_id_property=current_values.get(MemgraphParams.NODE_ID_PROPERTY, ""),
        num_epochs=current_values.get(TrainParams.NUM_EPOCHS, 0),
        console_log_freq=current_values.get(TrainParams.CONSOLE_LOG_FREQ, 0),
        checkpoint_freq=current_values.get(TrainParams.CHECKPOINT_FREQ, 0),
        device_type=current_values.get(OtherParams.DEVICE_TYPE, ""),
        path_to_model=current_values.get(OtherParams.PATH_TO_MODEL, ""),
    )
    return computed_return_value


set_model_parameters.__annotations__["return"] = mgp_Record(
    hidden_features_size=list,
    layer_type=str,
    aggregator=str,
    learning_rate=float,
    weight_decay=float,
    split_ratio=float,
    metrics=mgp_Any,
    node_id_property=str,
    num_epochs=int,
    console_log_freq=int,
    checkpoint_freq=int,
    device_type=str,
    path_to_model=str,
)
set_model_parameters = mgp_read_proc(set_model_parameters)


def checkpoint_sort_key(folder: str, file_name: str):
    """Return a chronological sort key (timestamp, sequence, filename) for a saved checkpoint, or False."""
    full_path = os_path.join(folder, file_name)
    if not os_path.isfile(full_path):
        return False
    match = CHECKPOINT_NAME.fullmatch(file_name)
    if match:
        seq = int(match.group("seq") or 0)
        return (match.group("saved_at"), seq, file_name)
    legacy = LEGACY_CHECKPOINT_NAME.fullmatch(file_name)
    if legacy:
        parts = legacy.group("saved_at").split("-")
        canonical_ts = f"{parts[0]}{parts[1]}{parts[2]}T{parts[3]}{parts[4]}{parts[5]}000000Z"
        seq = int(legacy.group("seq") or 0)
        return (canonical_ts, seq, file_name)
    if file_name.startswith("model") and file_name.endswith(".pt"):
        mtime_dt = datetime.fromtimestamp(os_path.getmtime(full_path), tz=UTC)
        return (mtime_dt.strftime(CHECKPOINT_TIME_FORMAT), 0, file_name)
    return False


def fetch_saved_models():
    """Return (model_saving_folder, models_newest_first) sorted chronologically across architectures."""
    model_saving_folder = os_path.join(
        os_getcwd(),
        current_values.get(OtherParams.MODEL_SAVING_FOLDER, DEFAULT_VALUES[OtherParams.MODEL_SAVING_FOLDER]),
    )
    if not os_path.isdir(model_saving_folder):
        return model_saving_folder, []
    saved = []
    for file_name in os_listdir(model_saving_folder):
        key = checkpoint_sort_key(model_saving_folder, file_name)
        if key:
            saved.append(key)
    models = [entry[2] for entry in sorted(saved, reverse=True)]
    return model_saving_folder, models


def save_model_to_folder() -> str:
    """Atomically write a new checkpoint before pruning older checkpoints beyond max_models_to_keep."""
    global checkpoint_sequence
    model_saving_folder = declare_saving_paths(current_values)
    layer_type = current_values.get(ModelParams.LAYER_TYPE, "GATJK")

    while True:
        checkpoint_sequence += 1
        saved_at = datetime.now(UTC).strftime(CHECKPOINT_TIME_FORMAT)
        file_name = f"model_{layer_type}_{saved_at}_{checkpoint_sequence:06d}.pt"
        path_to_saved_model = os_path.join(model_saving_folder, file_name)
        if not os_path.exists(path_to_saved_model):
            break

    checkpoint = {key: deepcopy(current_values.get(key, default)) for key, default in CHECKPOINT_TASK_FIELDS.items()}
    checkpoint[CHECKPOINT_STATE_DICT] = model.state_dict()

    descriptor, temporary_path = mkstemp(dir=model_saving_folder, prefix=".checkpoint-", suffix=".tmp")
    published = False
    try:
        with os_fdopen(descriptor, "wb") as checkpoint_file:
            torch_save(checkpoint, checkpoint_file)
            checkpoint_file.flush()
            os_fsync(checkpoint_file.fileno())
        os_replace(temporary_path, path_to_saved_model)
        published = True
    finally:
        if not published and os_path.exists(temporary_path):
            os_remove(temporary_path)

    _, models = fetch_saved_models()
    keep = max(1, int(current_values.get(TrainParams.MAX_MODELS_TO_KEEP, DEFAULT_VALUES[TrainParams.MAX_MODELS_TO_KEEP])))
    for superseded in models[keep:]:
        superseded_path = os_path.join(model_saving_folder, superseded)
        if os_path.exists(superseded_path):
            os_remove(superseded_path)

    return path_to_saved_model


def train(ctx: mgp_ProcCtx, num_epochs: mgp_Nullable[int] = None) -> list:
    """Train a fresh node classifier and return this run's logged epoch metrics."""
    global model, current_values, logged_data

    if not current_values:
        current_values = deepcopy(DEFAULT_VALUES)

    if num_epochs is None:
        effective_epochs = int(current_values.get(TrainParams.NUM_EPOCHS, DEFAULT_VALUES[TrainParams.NUM_EPOCHS]))
    else:
        if isinstance(num_epochs, bool) or not isinstance(num_epochs, int) or num_epochs < 1:
            raise ValueError(f"{TrainParams.NUM_EPOCHS} must be a positive integer")
        effective_epochs = num_epochs

    candidate = deepcopy(current_values)
    data = declare_data_into(ctx, candidate)
    trained_model = hetero_network_from_values(candidate)
    opt = torch_optim.Adam(
        trained_model.parameters(),
        lr=candidate.get(OptimizerParams.LEARNING_RATE, 0.1),
        weight_decay=candidate.get(OptimizerParams.WEIGHT_DECAY, 5e-4),
    )
    criterion = torch_nn.CrossEntropyLoss()

    candidate[TrainParams.NUM_EPOCHS] = effective_epochs
    current_values = candidate
    model = trained_model
    logged_data = []

    num_nodes_sample = current_values.get(HeteroParams.NUM_NODES_SAMPLE, 512)
    num_iterations_sample = current_values.get(HeteroParams.NUM_ITERATIONS_SAMPLE, 4)

    last_loss = float("inf")
    trigger_times = 0
    last_time = time()

    for epoch in tqdm(range(1, effective_epochs + 1)):
        loss, val_loss = train_epoch(
            model,
            opt,
            data,
            criterion,
            current_values.get(TrainParams.BATCH_SIZE, 64),
            current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, ""),
            {key: [num_nodes_sample] * num_iterations_sample for key in data.node_types},
        )

        if val_loss > last_loss:
            trigger_times += 1
            if trigger_times >= current_values.get(OtherParams.PATIENCE, 10):
                break
        else:
            trigger_times = 0

        last_loss = val_loss

        if epoch % current_values.get(TrainParams.CONSOLE_LOG_FREQ, 5) == 0:
            model.eval()
            with torch_no_grad():
                out = model(data.x_dict, data.edge_index_dict)
            observed_attr = current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")
            dict_train = metrics(
                data[observed_attr].train_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, []),
                observed_attr,
                current_values.get(OtherParams.DEVICE_TYPE, "cpu"),
            )
            dict_val = metrics(
                data[observed_attr].val_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, []),
                observed_attr,
                current_values.get(OtherParams.DEVICE_TYPE, "cpu"),
            )
            logged_data.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "val_loss": val_loss,
                    "train": dict_train,
                    "val": dict_val,
                }
            )
            summary = f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
            if "accuracy" in dict_train and "accuracy" in dict_val:
                summary += f", Accuracy: {dict_train['accuracy']:.4f}, Val Accuracy: {dict_val['accuracy']:.4f}"
            print(summary)

        if epoch % current_values.get(TrainParams.CHECKPOINT_FREQ, 5) == 0:
            if time() - last_time > current_values.get(TrainParams.TIME_BETWEEN_CHECKPOINTS, 2.0):
                save_model_to_folder()
                last_time = time()

    computed_return_value = [
        mgp_Record(
            epoch=entry.get("epoch", 0),
            loss=entry.get("loss", 0.0),
            val_loss=entry.get("val_loss", 0.0),
            train_log=entry.get("train", {}),
            val_log=entry.get("val", {}),
        )
        for entry in logged_data
    ]
    return computed_return_value


train.__annotations__["return"] = mgp_Record(epoch=int, loss=float, val_loss=float, train_log=mgp_Any, val_log=mgp_Any)
train = mgp_read_proc(train)


def get_training_data() -> list:
    """Return logged training epochs from the most recent training run."""
    computed_return_value = [
        mgp_Record(
            epoch=entry.get("epoch", 0),
            loss=entry.get("loss", 0.0),
            val_loss=entry.get("val_loss", 0.0),
            train_log=entry.get("train", {}),
            val_log=entry.get("val", {}),
        )
        for entry in logged_data
    ]
    return computed_return_value


get_training_data.__annotations__["return"] = mgp_Record(epoch=int, loss=float, val_loss=float, train_log=mgp_Any, val_log=mgp_Any)
get_training_data = mgp_read_proc(get_training_data)


def save_model() -> mgp_Record:
    """Save the active model and task metadata to model_saving_folder."""
    if model is False or model is None:
        raise RuntimeError("There are no initialized or loaded models. First load or initialize a model to be able save it.")

    path_to_saved_model = save_model_to_folder()
    computed_return_value = mgp_Record(path=path_to_saved_model, status="Model has been successfully saved.")
    return computed_return_value


save_model.__annotations__["return"] = mgp_Record(path=str, status=str)
save_model = mgp_read_proc(save_model)


def load_model(num: int = 0) -> mgp_Record:
    """Atomically load checkpoint num (newest first across architectures) without clobbering active state on failure."""
    global model, current_values

    if not current_values:
        current_values = deepcopy(DEFAULT_VALUES)

    model_saving_folder, models = fetch_saved_models()
    if len(models) == 0:
        raise FileNotFoundError("There are no saved models. ")

    if not -len(models) <= num < len(models):
        raise ValueError(f"Model with number {num} does not exist. There are {len(models)} models saved.")

    path_to_load_model = os_path.join(model_saving_folder, models[num])
    device = resolve_device(current_values.get(OtherParams.DEVICE_TYPE, "cpu"))
    checkpoint = torch_load(path_to_load_model, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path_to_load_model} is not a checkpoint dictionary.")
    missing = [key for key in (*CHECKPOINT_TASK_FIELDS, CHECKPOINT_STATE_DICT) if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint {path_to_load_model} does not record {', '.join(missing)}.")

    candidate = deepcopy(current_values)
    candidate.update({key: deepcopy(checkpoint.get(key, default)) for key, default in CHECKPOINT_TASK_FIELDS.items()})
    candidate[OtherParams.DEVICE_TYPE] = device
    loaded_model = hetero_network_from_values(candidate)
    loaded_model.load_state_dict(checkpoint.get(CHECKPOINT_STATE_DICT, {}))

    model = loaded_model
    current_values = candidate

    computed_return_value = mgp_Record(path=path_to_load_model, status="Model has been successfully loaded.")
    return computed_return_value


load_model.__annotations__["return"] = mgp_Record(path=str, status=str)
load_model = mgp_read_proc(load_model)


def predict(ctx: mgp_ProcCtx, vertex: mgp_Vertex) -> mgp_Record:
    """Predict the class of vertex using the trained model's preserved class vocabulary and device."""
    if model is False or model is None:
        raise RuntimeError("Load a model before predicting.")

    observed_attribute = current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")
    device = resolve_device(current_values.get(OtherParams.DEVICE_TYPE, "cpu"))
    nodes = list(iter(ctx.graph.vertices))
    if not nodes:
        raise ValueError("Graph is empty.")
    data, _, _, inv_reindexing, _, _ = extract_from_database(
        nodes,
        current_values.get(DataParams.SPLIT_RATIO, 0.8),
        current_values.get(HeteroParams.FEATURES_NAME, "features"),
        current_values.get(HeteroParams.CLASS_NAME, "class"),
        device,
        observed_attribute,
    )

    missing_node_types = set(current_values.get(HeteroParams.NODE_TYPES, [])) - set(data.node_types)
    missing_edge_types = set(current_values.get(HeteroParams.EDGE_TYPES, [])) - set(data.edge_types)
    if missing_node_types or missing_edge_types:
        raise ValueError(
            f"Graph lacks node types {sorted(missing_node_types)} / edge types {sorted(missing_edge_types)} the model uses."
        )
    feature_width = data[observed_attribute].x.size(dim=1)
    if feature_width != current_values.get(ModelParams.IN_CHANNELS, 0):
        raise ValueError(
            f"Features have width {feature_width}; the model expects {current_values.get(ModelParams.IN_CHANNELS, 0)}."
        )

    position = inv_reindexing.get(observed_attribute, {}).get(vertex.id, -1)
    if position < 0:
        raise ValueError(f"Vertex {vertex.id} is not a {observed_attribute} node with features.")

    model.eval()
    with torch_no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    if observed_attribute not in out:
        raise KeyError(f"Model produced no output for node type {observed_attribute!r}.")
    pred = out.get(observed_attribute, torch_zeros((0, 0))).argmax(dim=1)

    predicted_index = int(pred.detach().cpu().numpy()[position])
    inv_label_reindexing = current_values.get(HeteroParams.INV_LABEL_REINDEXING, {})
    if predicted_index not in inv_label_reindexing:
        raise ValueError(f"Output column {predicted_index} has no class in the model's vocabulary.")

    computed_return_value = mgp_Record(
        predicted_class=inv_label_reindexing.get(predicted_index, 0),
        status="Prediction complete.",
    )
    return computed_return_value


predict.__annotations__["return"] = mgp_Record(predicted_class=int, status=str)
predict = mgp_read_proc(predict)


def reset() -> mgp_Record:
    """Reset model, logged_data, and current_values to fresh defaults."""
    global model, current_values, logged_data
    model = False
    logged_data = []
    current_values = deepcopy(DEFAULT_VALUES)

    computed_return_value = mgp_Record(status="Global parameters and logged data have been reset")
    return computed_return_value


reset.__annotations__["return"] = mgp_Record(status=str)
reset = mgp_read_proc(reset)
