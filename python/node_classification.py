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
from tempfile import mkstemp
from time import time

from mgp import (
    Any as mgp_Any,
    List as mgp_List,
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

##############################
# constants
##############################


# parameters for the model
DEFAULT_ARGUMENT_DICT = {}


class ModelParams:
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    HIDDEN_FEATURES_SIZE = "hidden_features_size"
    LAYER_TYPE = "layer_type"
    AGGREGATOR = "aggregator"


# parameters for optimizer
class OptimizerParams:
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"


# parameters for data
class DataParams:
    SPLIT_RATIO = "split_ratio"
    METRICS = "metrics"


# parameters relevant to memgraph database
class MemgraphParams:
    NODE_ID_PROPERTY = "node_id_property"


# parameters for training
class TrainParams:
    NUM_EPOCHS = "num_epochs"
    CONSOLE_LOG_FREQ = "console_log_freq"
    CHECKPOINT_FREQ = "checkpoint_freq"
    BATCH_SIZE = "batch_size"
    MAX_MODELS_TO_KEEP = "max_models_to_keep"
    TIME_BETWEEN_CHECKPOINTS = "time_between_checkpoints"


# parameters relevant for heterogeneous structure
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


# other necessary parameters
class OtherParams:
    DEVICE_TYPE = "device_type"
    PATH_TO_MODEL = "path_to_model"
    PATIENCE = "patience"
    MODEL_SAVING_FOLDER = "model_saving_folder"


GAT_MODEL = "GAT"
GATV2_MODEL = "GATv2"
SAGE_MODEL = "SAGE"
GAT_WITH_JK = "GATJK"

# dictionary of models
MODELS = {GAT_MODEL: GAT, GATV2_MODEL: GATv2, SAGE_MODEL: SAGE, GAT_WITH_JK: GATJK}

global model, current_values

model: mgp_Any = False
current_values: dict = {}

# list for saving logged data
logged_data: mgp_List = []

# dictionary of defined input types
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

# dictionary of default values for input types
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
    OtherParams.MODEL_SAVING_FOLDER: "/tmp/torch_models",
}

# numeric parameters and their admitted domains: frequencies, retention and sizes are
# divisors, counts or slice bounds, so they must be positive integers
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
NON_NEGATIVE_FLOAT_PARAMETERS = (OptimizerParams.WEIGHT_DECAY, TrainParams.TIME_BETWEEN_CHECKPOINTS)

# What a checkpoint records next to its weights, so a loaded model keeps the
# architecture it was trained with and the meaning of each output column. Each
# field maps to the false value of its type; the values are read-only defaults.
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
# model_<layer type>_<UTC save time to the microsecond>.pt; the fixed-width time sorts chronologically
CHECKPOINT_TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"
CHECKPOINT_NAME = re_compile(r"model_(?P<layer_type>[A-Za-z0-9]+)_(?P<saved_at>\d{8}T\d{12}Z)\.pt")


##############################
# set model parameters
##############################


def declare_data(ctx: mgp_ProcCtx) -> HeteroData:
    """This function initializes global variable data.

    Args:
        ctx (mgp.ProcCtx): current context
    """
    global current_values

    # change device type to cuda if possible
    current_values[OtherParams.DEVICE_TYPE] = available_device()

    nodes = list(iter(ctx.graph.vertices))  # obtain nodes from context
    if not nodes:
        raise Exception("Graph is empty.")

    # extraction of data from database to torch.Tensors
    (
        data,
        current_values[HeteroParams.OBSERVED_ATTRIBUTE],
        current_values[HeteroParams.REINDEXING],
        current_values[HeteroParams.INV_REINDEXING],
        current_values[HeteroParams.LABEL_REINDEXING],
        current_values[HeteroParams.INV_LABEL_REINDEXING],
    ) = extract_from_database(
        nodes,
        current_values.get(DataParams.SPLIT_RATIO, 0.0),
        current_values.get(HeteroParams.FEATURES_NAME, ""),
        current_values.get(HeteroParams.CLASS_NAME, ""),
        current_values.get(OtherParams.DEVICE_TYPE, ""),
    )

    observed_attribute = current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")
    if observed_attribute not in data.node_types:
        raise ValueError(f"No node with property '{current_values.get(HeteroParams.CLASS_NAME, '')}' has features.")
    # node stores are selected by subscription; HeteroData.get reads the global store
    observed_attribute_data = data[observed_attribute]

    # second parameter of shape of feature matrix is number of input channels
    current_values[ModelParams.IN_CHANNELS] = observed_attribute_data.x.size(dim=1)

    # one output column per class of the task vocabulary
    current_values[ModelParams.OUT_CHANNELS] = len(current_values.get(HeteroParams.LABEL_REINDEXING, {}))

    # graph metadata the heterogeneous model is built for
    current_values[HeteroParams.NODE_TYPES] = list(data.node_types)
    current_values[HeteroParams.EDGE_TYPES] = list(data.edge_types)

    return data


def available_device() -> str:
    device = "cuda:0" if torch_cuda.is_available() else "cpu"
    return device


def hetero_network_from_values(values: dict):
    """The heterogeneous network that values describe: architecture, task width, graph metadata and device.

    Fresh training and checkpoint restore both go through this one translation, so a
    loaded model always has the architecture it was trained with.

    Args:
        values (dict): configuration and task values, as in current_values or a checkpoint
    """

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

    # choose model architecture according to layer type
    layer_type = values.get(ModelParams.LAYER_TYPE, "")

    if layer_type not in MODELS:
        raise Exception(
            "You didn't choose one of currently available models (GAT, GATv2, GATJK and SAGE). Please choose one of them."
        )

    args = args_gatjk if layer_type == GAT_WITH_JK else args_inductive

    network = MODELS.get(layer_type, GAT)(*args)

    # convert model to hetero structure
    # (if graph is homogeneous, we also do this conversion since all calculations are same)
    metadata = (values.get(HeteroParams.NODE_TYPES, []), values.get(HeteroParams.EDGE_TYPES, []))
    network = to_hetero(network, metadata)

    # move model to device
    network.to(values.get(OtherParams.DEVICE_TYPE, ""))

    return network


def declare_saving_paths(values: dict) -> None:
    """This function creates the model saving folder and records the checkpoint path prefix in values."""
    # either make new folder for saving models, or use existing one with exactly this name
    path = os_path.join(os_getcwd(), values.get(OtherParams.MODEL_SAVING_FOLDER, ""))
    try:
        os_makedirs(path)
        print(f"New folder for saving models was created on destination {path}.")
    except FileExistsError:
        print(f"Folder for saving models already exists on destination {path}.")

    values[OtherParams.PATH_TO_MODEL] = os_path.join(path, "model_" + values.get(ModelParams.LAYER_TYPE, "") + "_")


def configuration_errors(values: dict) -> list[str]:
    """Every declared numeric/list domain that values violate; the type check runs first."""
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
    # both the training and the validation partition must be able to hold nodes
    split_ratio = values.get(DataParams.SPLIT_RATIO, 0.0)
    if not isfinite(split_ratio) or not 0 < split_ratio < 1:
        errors.append(f"{DataParams.SPLIT_RATIO} must be a finite number strictly between 0 and 1")
    hidden_features_size = values.get(ModelParams.HIDDEN_FEATURES_SIZE, [])
    if not hidden_features_size or any(
        not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in hidden_features_size
    ):
        errors.append(f"{ModelParams.HIDDEN_FEATURES_SIZE} must be a non-empty list of positive integers")
    return errors


@mgp_read_proc
def set_model_parameters(
    params: mgp_Any = DEFAULT_ARGUMENT_DICT,
) -> mgp_Record(
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
):
    """The purpose of this function is to initialize all global variables.
    _You_ can change those via **params** dictionary.
    It checks if variables in **params** are defined appropriately. If so,
    map of default global parameters is overridden with user defined dictionary params.
    After that it prepares the model saving folder and publishes the admitted configuration.

    Args:
        ctx: (mgp.ProcCtx): current context,
        params: (mgp.Map, optional): user defined parameters from query module. Defaults to {}

    Raises:
        Exception: exception is raised if some variable in dictionary params is not
                    defined as it should be

    Returns:
    mgp.Record(
        hidden_features_size (list): list of hidden features
        layer_type (str): type of layer
        aggregator (str): type of aggregator
        learning_rate (float): learning rate
        weight_decay (float): weight decay
        split_ratio (float): ratio between training and validation data
        metrics (list): list of metrics to be calculated
        node_id_property (str): name of nodes id property
        num_epochs (int): number of epochs
        console_log_freq (int): frequency of logging metrics
        checkpoint_freq (int): frequency of saving models
        device_type (str): cpu or cuda
        path_to_model (str): path where model is load and saved
    )
    """
    if params is DEFAULT_ARGUMENT_DICT:
        params = DEFAULT_ARGUMENT_DICT.copy()
    global DEFINED_INPUT_TYPES, DEFAULT_VALUES, current_values

    # override any default parameters in an isolated candidate; the defaults
    # (including their nested lists and maps) are never shared with run state
    candidate = deepcopy(DEFAULT_VALUES)
    candidate.update(params)

    # hidden_features_size and metrics are sometimes translated as tuples,
    # which are not hashable, but conversion to lists makes them hashable
    for list_parameter in (ModelParams.HIDDEN_FEATURES_SIZE, DataParams.METRICS):
        if isinstance(candidate.get(list_parameter, []), tuple):
            candidate[list_parameter] = list(candidate.get(list_parameter, []))

    # raise exception if some variable in dictionary params is not defined as it should be:
    # every defined parameter must be present and hold its declared type
    correctly_typed = all(
        name in candidate and isinstance(candidate.get(name, expected_type()), expected_type)
        for name, expected_type in DEFINED_INPUT_TYPES.items()
    )
    if not correctly_typed:
        raise Exception("Input dictionary is not correctly typed.")
    errors = configuration_errors(candidate)
    if errors:
        raise ValueError("Input dictionary is out of range: " + "; ".join(errors))

    # define paths, then publish the admitted configuration
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


##############################
# train
##############################


def fetch_saved_models():
    """The purpose of this function is to fetch the saved checkpoints of the configured layer type.

    Returns:
        model_saving_folder (str): path to folder with saved models
        models (list): checkpoint file names of the configured layer type, newest save time first
    """
    model_saving_folder = os_path.join(os_getcwd(), current_values.get(OtherParams.MODEL_SAVING_FOLDER, ""))
    layer_type = current_values.get(ModelParams.LAYER_TYPE, "")
    saved = []
    for file_name in os_listdir(model_saving_folder):
        name = CHECKPOINT_NAME.fullmatch(file_name)
        if name and name.group("layer_type") == layer_type and os_path.isfile(os_path.join(model_saving_folder, file_name)):
            saved.append((name.group("saved_at"), file_name))

    models = [file_name for _, file_name in sorted(saved, reverse=True)]
    return model_saving_folder, models


def save_model_to_folder() -> str:
    """The purpose of this function is to save model to folder.

    The checkpoint is written to a unique temporary file, flushed to disk and
    atomically renamed into place; only then are older checkpoints of the same
    layer type beyond max_models_to_keep removed.

    Returns:
        path_to_saved_model (str): path to saved model
    """
    model_saving_folder = os_path.join(os_getcwd(), current_values.get(OtherParams.MODEL_SAVING_FOLDER, ""))
    layer_type = current_values.get(ModelParams.LAYER_TYPE, "")
    saved_at = datetime.now(UTC).strftime(CHECKPOINT_TIME_FORMAT)
    path_to_saved_model = os_path.join(model_saving_folder, f"model_{layer_type}_{saved_at}.pt")
    if os_path.exists(path_to_saved_model):
        raise FileExistsError(f"Checkpoint {path_to_saved_model} already exists.")

    checkpoint = {key: current_values.get(key, default) for key, default in CHECKPOINT_TASK_FIELDS.items()}
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
        if not published:
            os_remove(temporary_path)

    # delete oldest models if there are more than max models to keep
    _, models = fetch_saved_models()
    for superseded in models[current_values.get(TrainParams.MAX_MODELS_TO_KEEP, 0) :]:
        os_remove(os_path.join(model_saving_folder, superseded))

    return path_to_saved_model


@mgp_read_proc
def train(
    ctx: mgp_ProcCtx, num_epochs: int = 100
) -> mgp_Record(epoch=int, loss=float, val_loss=float, train_log=mgp_Any, val_log=mgp_Any):
    """This function performs training of model. It first declares data, model,
    optimizer and criterion. Then it performs training.

    Args:
        ctx (mgp.ProcCtx): context of process
        num_epochs (int, optional): number of epochs. Defaults to 100.

    Raises:
        Exception: raised if graph is empty

    Returns:
        list of mgp.Record of
        epoch (int): epoch number
        loss (float): loss of model on training data
        val_loss (float): loss of model on validation data
        train_log (list): list of metrics on training data
        val_log (list): list of metrics on validation data
    """
    global model, current_values, logged_data

    if isinstance(num_epochs, bool) or num_epochs < 1:
        raise ValueError(f"{TrainParams.NUM_EPOCHS} must be a positive integer")

    # define fresh data
    data = declare_data(ctx)

    # define model, optimizer and criterion
    model = hetero_network_from_values(current_values)
    opt = torch_optim.Adam(
        model.parameters(),
        lr=current_values.get(OptimizerParams.LEARNING_RATE, 0.0),
        weight_decay=current_values.get(OptimizerParams.WEIGHT_DECAY, 0.0),
    )
    criterion = torch_nn.CrossEntropyLoss()

    current_values[TrainParams.NUM_EPOCHS] = num_epochs
    num_nodes_sample = current_values.get(HeteroParams.NUM_NODES_SAMPLE, 0)
    num_iterations_sample = current_values.get(HeteroParams.NUM_ITERATIONS_SAMPLE, 0)

    # variables for early stopping
    last_loss = float("inf")
    trigger_times = 0
    last_time = time()
    # training
    for epoch in tqdm(range(1, num_epochs + 1)):
        # one epoch of training, both training and validation loss are returned
        loss, val_loss = train_epoch(
            model,
            opt,
            data,
            criterion,
            current_values.get(TrainParams.BATCH_SIZE, 0),
            current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, ""),
            {key: [num_nodes_sample] * num_iterations_sample for key in data.node_types},
        )

        # early stopping
        if val_loss > last_loss:
            trigger_times += 1

            drop_epochs = str(trigger_times) + " " + ("consecutive epochs" if trigger_times > 1 else "consecutive epoch")

            times_until_stopping = current_values.get(OtherParams.PATIENCE, 0) - trigger_times

            stop_after = str(times_until_stopping) + " " + ("more drops" if times_until_stopping > 1 else "more drop")

            print(f"Loss has dropped for {drop_epochs}. Stopping after {stop_after}.")

            if trigger_times >= current_values.get(OtherParams.PATIENCE, 0):
                print("Early stopping!")
                break

        else:
            trigger_times = 0

        last_loss = val_loss

        # log data every console_log_freq epochs
        if epoch % current_values.get(TrainParams.CONSOLE_LOG_FREQ, 0) == 0:
            model.eval()
            with torch_no_grad():
                out = model(data.x_dict, data.edge_index_dict)
            dict_train = metrics(
                data[current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")].train_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, []),
                current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, ""),
                current_values.get(OtherParams.DEVICE_TYPE, ""),
            )
            dict_val = metrics(
                data[current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")].val_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, []),
                current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, ""),
                current_values.get(OtherParams.DEVICE_TYPE, ""),
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

            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f},"
                + (
                    f"Accuracy: {logged_data[-1].get('train', {}).get('accuracy', 0.0):.4f}, Accuracy: "
                    f"{logged_data[-1].get('val', {}).get('accuracy', 0.0):.4f}"
                )
            )

        # save model every checkpoint_freq epochs
        if epoch % current_values.get(TrainParams.CHECKPOINT_FREQ, 0) == 0:
            if time() - last_time > current_values.get(TrainParams.TIME_BETWEEN_CHECKPOINTS, 0.0):
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


##############################
# get training data
##############################


@mgp_read_proc
def get_training_data() -> mgp_Record(epoch=int, loss=float, val_loss=float, train_log=mgp_Any, val_log=mgp_Any):
    """This function is used so user can see what is logged data from training.


    Returns:
        mgp.Record(
            epoch (int): epoch number of record of logged data row
            loss (float): loss in logged data row
            val_loss (float): validation loss in logged data row
            train_log (mgp.Any): training parameters of record of logged data row
            val_log (mgp.Any): validation parameters of record of logged data row
            ): record to return


    """

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


##############################
# model loading and saving, predict
##############################


@mgp_read_proc
def save_model() -> mgp_Record(path=str, status=str):
    """This function saves model to model saving folder. If there are already total
    of max_models_to_keep models in model saving folder, oldest model is deleted.

    Exception: raised if model is not initialized or defined

    Returns:
        mgp.Record(
            path (str): path to saved model
            status (str): status of saving model
            ): return record
    """

    if model is False:
        raise Exception("There are no initialized or loaded models. First load or initialize a model to be able save it.")

    path_to_saved_model = save_model_to_folder()

    computed_return_value = mgp_Record(path=path_to_saved_model, status="Model has been successfully saved.")
    return computed_return_value


@mgp_read_proc
def load_model(num: int = 0) -> mgp_Record(path=str, status=str):
    """This function loads model from defined folder for saved models.

    The checkpoint's own architecture, graph metadata, observed node type,
    class vocabulary and feature/class property names are restored with its
    weights, so predictions keep the meaning the model was trained with.

    Args:
        num (int, optional): ordinary number of model of the configured layer type to load,
            newest first. Defaults to 0 (newest model).

    Returns:
        mgp.Record(path (str): path to loaded model): return record
    """
    global model, current_values

    model_saving_folder, models = fetch_saved_models()

    if len(models) == 0:
        raise Exception("There are no saved models.")

    if not -len(models) <= num < len(models):
        raise Exception(f"Model with number {num} does not exist. There are {len(models)} models saved.")

    path_to_load_model = os_path.join(model_saving_folder, models[num])

    device = available_device()
    checkpoint = torch_load(path_to_load_model, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path_to_load_model} is not a checkpoint dictionary.")
    missing = [key for key in (*CHECKPOINT_TASK_FIELDS, CHECKPOINT_STATE_DICT) if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint {path_to_load_model} does not record {', '.join(missing)}.")

    # build and fill the model on an isolated copy; publish both only after the weights load
    candidate = deepcopy(current_values)
    candidate.update({key: checkpoint.get(key, default) for key, default in CHECKPOINT_TASK_FIELDS.items()})
    candidate[OtherParams.DEVICE_TYPE] = device
    loaded_model = hetero_network_from_values(candidate)
    loaded_model.load_state_dict(checkpoint.get(CHECKPOINT_STATE_DICT, {}))

    model = loaded_model
    current_values = candidate

    computed_return_value = mgp_Record(path=path_to_load_model, status="Model has been successfully loaded.")
    return computed_return_value


@mgp_read_proc
def predict(ctx: mgp_ProcCtx, vertex: mgp_Vertex) -> mgp_Record(predicted_class=int, status=str):
    """This function predicts metrics on one node. It is suggested that user previously
    loads unseen test data to predict on it.

    Example of usage:
        MATCH (n {id: 1}) CALL node_classification.predict(n) YIELD * RETURN predicted_class;

        # note: if node with property id = 1 doesn't exist, query module won't be called

    Args:
        ctx (mgp.ProcCtx): proc context
        vertex (mgp.Vertex): node to predict on

    Returns:
        mgp.Record(
            predicted_class (int): predicted class
            status (str): status of prediction
        ): record to return
    """
    if model is False:
        raise Exception("Load a model before predicting.")

    # The graph is read with the trained task's observed type and property names;
    # its own labels (if any) never redefine the model's output columns.
    observed_attribute = current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")
    nodes = list(iter(ctx.graph.vertices))
    if not nodes:
        raise Exception("Graph is empty.")
    data, _, _, inv_reindexing, _, _ = extract_from_database(
        nodes,
        current_values.get(DataParams.SPLIT_RATIO, 0.0),
        current_values.get(HeteroParams.FEATURES_NAME, ""),
        current_values.get(HeteroParams.CLASS_NAME, ""),
        current_values.get(OtherParams.DEVICE_TYPE, ""),
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
    # the heterogeneous model returns one logits matrix per node type it was built for
    if observed_attribute not in out:
        raise KeyError(f"Model produced no output for node type {observed_attribute!r}.")
    pred = out.get(observed_attribute, torch_zeros((0, 0))).argmax(dim=1)

    predicted_index = int(pred.detach().cpu().numpy()[position])

    # output columns decode through the trained task's class vocabulary
    inv_label_reindexing = current_values.get(HeteroParams.INV_LABEL_REINDEXING, {})
    if predicted_index not in inv_label_reindexing:
        raise ValueError(f"Output column {predicted_index} has no class in the model's vocabulary.")

    computed_return_value = mgp_Record(
        predicted_class=inv_label_reindexing.get(predicted_index, 0),
        status="Prediction complete.",
    )
    return computed_return_value


@mgp_read_proc
def reset() -> mgp_Record(status=str):
    """This function resets all variables to default values.

    Returns:
        mgp.Record(status (str): status of reset): record to return
    """

    # set model and logged_data to None
    global model, current_values, logged_data
    model = False
    logged_data = []

    # reinitialize current_values with a copy, so later run state never writes into the defaults
    current_values = deepcopy(DEFAULT_VALUES)

    computed_return_value = mgp_Record(status="Global parameters and logged data have been reset")
    return computed_return_value
