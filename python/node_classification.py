"""Utilities for node classification."""

from datetime import datetime
from os import getcwd as os_getcwd
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os import path as os_path
from os import remove as os_remove
from time import time

from mgp import Any as mgp_Any
from mgp import List as mgp_List
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import Vertex as mgp_Vertex
from mgp import read_proc as mgp_read_proc
from torch import cuda as torch_cuda
from torch import device as torch_device
from torch import load as torch_load
from torch import nn as torch_nn
from torch import optim as torch_optim
from torch import save as torch_save
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
    current_values[OtherParams.DEVICE_TYPE] = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")

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
        current_values.get(DataParams.SPLIT_RATIO, False),
        current_values.get(HeteroParams.FEATURES_NAME, False),
        current_values.get(HeteroParams.CLASS_NAME, False),
        current_values.get(OtherParams.DEVICE_TYPE, False),
    )

    observed_attribute_data = data.get(current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False), {})

    # second parameter of shape of feature matrix is number of input channels
    current_values[ModelParams.IN_CHANNELS] = observed_attribute_data.x.size(dim=1)

    # number of output channels is number of classes in the dataset
    current_values[ModelParams.OUT_CHANNELS] = len(set(observed_attribute_data.y.detach().cpu().numpy()))

    return data


def declare_model(data: mgp_Any):
    """This function initializes global variables model, opt and criterion.

    Args:
        ctx (mgp.ProcCtx): current context
    """

    # choose one of the available layer types
    global model, current_values

    args_gatjk = [
        current_values.get(ModelParams.IN_CHANNELS, False),
        current_values.get(ModelParams.HIDDEN_FEATURES_SIZE, False),
        current_values.get(ModelParams.OUT_CHANNELS, False),
    ]

    args_inductive = [
        current_values.get(ModelParams.IN_CHANNELS, False),
        current_values.get(ModelParams.HIDDEN_FEATURES_SIZE, False),
        current_values.get(ModelParams.OUT_CHANNELS, False),
        current_values.get(ModelParams.AGGREGATOR, False),
    ]

    # choose model architecture according to layer type
    layer_type = current_values.get(ModelParams.LAYER_TYPE, "")

    if layer_type not in MODELS:
        raise Exception(
            "You didn't choose one of currently available models (GAT, GATv2, GATJK and SAGE). Please choose one of them."
        )

    args = args_gatjk if layer_type == GAT_WITH_JK else args_inductive

    model = MODELS.get(layer_type, GAT)(*args)

    # convert model to hetero structure
    # (if graph is homogeneous, we also do this conversion since all calculations are same)
    metadata = (data.node_types, data.edge_types)
    model = to_hetero(model, metadata)

    # move model to device
    model.to(current_values.get(OtherParams.DEVICE_TYPE, False))

    # set default optimizer
    opt = torch_optim.Adam(
        model.parameters(),
        lr=current_values.get(OptimizerParams.LEARNING_RATE, False),
        weight_decay=current_values.get(OptimizerParams.WEIGHT_DECAY, False),
    )

    # set default criterion
    criterion = torch_nn.CrossEntropyLoss()

    return opt, criterion


def declare_saving_paths():
    """This function initializes global variables paths."""
    global current_values
    # either make new folder for saving models, or use existing one with exactly this name
    try:
        path = os_path.join(os_getcwd(), current_values.get(OtherParams.MODEL_SAVING_FOLDER, False))
        os_makedirs(path)
        print(f"New folder for saving models was created on destination {path}.")
    except FileExistsError:
        print(f"Folder for saving models already exists on destination {path}.")

    current_values[OtherParams.PATH_TO_MODEL] = os_path.join(
        os_getcwd(),
        current_values.get(OtherParams.MODEL_SAVING_FOLDER, False),
        "model_" + current_values.get(ModelParams.LAYER_TYPE, "") + "_",
    )
    return False


@mgp_read_proc
def set_model_parameters(
    params: mgp_Any = DEFAULT_ARGUMENT_DICT,
) -> mgp_Record:
    """The purpose of this function is to initialize all global variables.
    _You_ can change those via **params** dictionary.
    It checks if variables in **params** are defined appropriately. If so,
    map of default global parameters is overridden with user defined dictionary params.
    After that it executes previously defined functions declare_globals and
    declare_model_and_data and sets each global variable to some value.

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

    # function checks if input values in dictionary are correctly typed
    def is_correctly_typed(defined_types, input_values):
        if isinstance(defined_types, dict) and isinstance(input_values, dict):
            # defined_types is a dict of types
            computed_return_value = all(
                k in input_values  # check if exists
                and is_correctly_typed(defined_types[k], input_values[k])  # check for correct type
                for k in defined_types
            )
            return computed_return_value
        elif isinstance(defined_types, type):
            computed_return_value = isinstance(input_values, defined_types)
            return computed_return_value
        else:
            return False

    # hidden_features_size and metrics are sometimes translated as tuples,
    # which are not hashable, but conversion to lists makes them hashable
    if ModelParams.HIDDEN_FEATURES_SIZE in params.keys() and isinstance(params[ModelParams.HIDDEN_FEATURES_SIZE], tuple):
        params[ModelParams.HIDDEN_FEATURES_SIZE] = list(params[ModelParams.HIDDEN_FEATURES_SIZE])
    if DataParams.METRICS in params.keys() and isinstance(params[DataParams.METRICS], tuple):
        params[DataParams.METRICS] = list(params[DataParams.METRICS])

    # override any default parameters
    current_values = {**DEFAULT_VALUES, **params}

    # raise exception if some variable in dictionary params is not defined as it should be
    if not is_correctly_typed(DEFINED_INPUT_TYPES, current_values):
        raise Exception("Input dictionary is not correctly typed.")

    # define paths
    declare_saving_paths()

    computed_return_value = mgp_Record(
        hidden_features_size=current_values.get(ModelParams.HIDDEN_FEATURES_SIZE, 0),
        layer_type=current_values.get(ModelParams.LAYER_TYPE, ""),
        aggregator=current_values.get(ModelParams.AGGREGATOR, False),
        learning_rate=current_values.get(OptimizerParams.LEARNING_RATE, 0.0),
        weight_decay=current_values.get(OptimizerParams.WEIGHT_DECAY, False),
        split_ratio=current_values.get(DataParams.SPLIT_RATIO, 0.0),
        metrics=current_values.get(DataParams.METRICS, False),
        node_id_property=current_values.get(MemgraphParams.NODE_ID_PROPERTY, False),
        num_epochs=current_values.get(TrainParams.NUM_EPOCHS, False),
        console_log_freq=current_values.get(TrainParams.CONSOLE_LOG_FREQ, False),
        checkpoint_freq=current_values.get(TrainParams.CHECKPOINT_FREQ, False),
        device_type=current_values.get(OtherParams.DEVICE_TYPE, ""),
        path_to_model=current_values.get(OtherParams.PATH_TO_MODEL, ""),
    )
    return computed_return_value


##############################
# train
##############################


def fetch_saved_models():
    """The purpose of this function is to fetch all saved models.

    Returns:
        model_saving_folder (str): path to folder with saved models
        models (list): list of paths of saved models
    """
    global model
    model_saving_folder = os_path.join(current_values.get(OtherParams.MODEL_SAVING_FOLDER, False))
    models = [
        f
        for f in os_listdir(model_saving_folder)
        if os_path.isfile(os_path.join(model_saving_folder, f)) and f.endswith(".pt") and f.startswith("model")
    ]

    models.sort(reverse=True)

    return model_saving_folder, models


def save_model_to_folder() -> str:
    """The purpose of this function is to save model to folder.

    Returns:
        path_to_saved_model (str): path to saved model
    """
    model_saving_folder, models = fetch_saved_models()

    # delete oldest models if there are more than max models to keep
    for i in range(current_values.get(TrainParams.MAX_MODELS_TO_KEEP, 0) - 1, len(models)):
        os_remove(os_path.join(model_saving_folder, models[i]))

    path_to_saved_model = current_values.get(OtherParams.PATH_TO_MODEL, 0.0) + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pt"
    torch_save(
        model.state_dict(),
        path_to_saved_model,
    )

    return path_to_saved_model


@mgp_read_proc
def train(ctx: mgp_ProcCtx, num_epochs: int = 100) -> list[mgp_Record]:
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

    # define fresh data
    data = declare_data(ctx)

    # define model
    opt, criterion = declare_model(data)

    current_values[TrainParams.NUM_EPOCHS] = num_epochs
    num_nodes_sample = current_values.get(HeteroParams.NUM_NODES_SAMPLE, False)
    num_iterations_sample = current_values.get(HeteroParams.NUM_ITERATIONS_SAMPLE, False)

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
            current_values.get(TrainParams.BATCH_SIZE, False),
            current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False),
            {key: [num_nodes_sample] * num_iterations_sample for key in data.node_types},
        )

        # early stopping
        if val_loss > last_loss:
            trigger_times += 1

            drop_epochs = str(trigger_times) + " " + ("consecutive epochs" if trigger_times > 1 else "consecutive epoch")

            times_until_stopping = current_values.get(OtherParams.PATIENCE, 0.0) - trigger_times

            stop_after = str(times_until_stopping) + " " + ("more drops" if times_until_stopping > 1 else "more drop")

            print(f"Loss has dropped for {drop_epochs}. Stopping after {stop_after}.")

            if trigger_times >= current_values.get(OtherParams.PATIENCE, 0):
                print("Early stopping!")
                break

        else:
            trigger_times = 0

        last_loss = val_loss

        # log data every console_log_freq epochs
        if epoch % current_values.get(TrainParams.CONSOLE_LOG_FREQ, 0.0) == 0:
            model.eval()
            out = model(data.x_dict, data.edge_index_dict)
            dict_train = metrics(
                data[current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")].train_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, False),
                current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False),
                current_values.get(OtherParams.DEVICE_TYPE, False),
            )
            dict_val = metrics(
                data[current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, "")].val_mask,
                out,
                data,
                current_values.get(DataParams.METRICS, False),
                current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False),
                current_values.get(OtherParams.DEVICE_TYPE, False),
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
                    f"Accuracy: {logged_data[-1].get('train', {}).get('accuracy', False):.4f}, Accuracy: "
                    f"{logged_data[-1].get('val', {}).get('accuracy', False):.4f}"
                )
            )

        # save model every checkpoint_freq epochs
        if epoch % current_values.get(TrainParams.CHECKPOINT_FREQ, 0.0) == 0:
            if time() - last_time > current_values.get(TrainParams.TIME_BETWEEN_CHECKPOINTS, 0.0):
                save_model_to_folder()
                last_time = time()

    computed_return_value = [
        mgp_Record(
            epoch=data.get("epoch", False),
            loss=data.get("loss", []),
            val_loss=data.get("val_loss", []),
            train_log=data.get("train", False),
            val_log=data.get("val", False),
        )
        for data in logged_data
    ]
    return computed_return_value


##############################
# get training data
##############################


@mgp_read_proc
def get_training_data() -> list[mgp_Record]:
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
            epoch=data.get("epoch", False),
            loss=data.get("loss", []),
            val_loss=data.get("val_loss", []),
            train_log=data.get("train", False),
            val_log=data.get("val", False),
        )
        for data in logged_data
    ]
    return computed_return_value


##############################
# model loading and saving, predict
##############################


@mgp_read_proc
def save_model() -> mgp_Record:
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
def load_model(ctx: mgp_ProcCtx, num: int = 0) -> mgp_Record:
    """This function loads model from defined folder for saved models.

    Args:
        num (int, optional): ordinary number of model to load from default map. Defaults to 0 (newest model).

    Returns:
        mgp.Record(path (str): path to loaded model): return record
    """
    global model

    data = declare_data(ctx)
    declare_model(data)

    model_saving_folder, models = fetch_saved_models()

    if len(models) == 0:
        raise Exception("There are no saved models.")

    if len(models) < (len(models) + num) % len(models) + 1:
        raise Exception(f"Model with number {num} does not exist. There are {len(models)} models saved.")

    path_to_load_model = os_path.join(model_saving_folder, models[num])

    model.load_state_dict(torch_load(path_to_load_model))

    computed_return_value = mgp_Record(path=path_to_load_model, status="Model has been successfully loaded.")
    return computed_return_value


@mgp_read_proc
def predict(ctx: mgp_ProcCtx, vertex: mgp_Vertex) -> mgp_Record:
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
    global current_values

    # define fresh data
    data = declare_data(ctx)

    if model is False:
        raise Exception("Load a model before predicting.")

    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out[current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False)].argmax(dim=1)

    inv_reindexing = HeteroParams.INV_REINDEXING
    observed_attribute = current_values.get(HeteroParams.OBSERVED_ATTRIBUTE, False)

    position = current_values.get(inv_reindexing, {})[observed_attribute][vertex.id]

    predicted_class = int(pred.detach().cpu().numpy()[position])

    computed_return_value = mgp_Record(
        predicted_class=current_values.get(HeteroParams.INV_LABEL_REINDEXING, {})[predicted_class],
        status="Prediction complete.",
    )
    return computed_return_value


@mgp_read_proc
def reset() -> mgp_Record:
    """This function resets all variables to default values.

    Returns:
        mgp.Record(status (str): status of reset): record to return
    """

    # set model and logged_data to None
    global model, current_values, logged_data
    model = False
    logged_data = []

    # reinitialize current_values
    current_values = DEFAULT_VALUES

    computed_return_value = mgp_Record(status="Global parameters and logged data have been reset")
    return computed_return_value
