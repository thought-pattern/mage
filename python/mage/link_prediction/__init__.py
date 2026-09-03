"""Public API for the link prediction package."""

from mage.link_prediction.constants import (
    Activations,  # noqa: F401
    Aggregators,  # noqa: F401
    Context,  # noqa: F401
    Devices,  # noqa: F401
    Metrics,  # noqa: F401
    Models,  # noqa: F401
    Optimizers,  # noqa: F401
    Parameters,  # noqa: F401
    Predictors,  # noqa: F401
    Reindex,  # noqa: F401
)
from mage.link_prediction.link_prediction_util import (
    add_self_loop,  # noqa: F401
    classify,  # noqa: F401
    inner_predict,  # noqa: F401
    inner_train,  # noqa: F401
    preprocess,  # noqa: F401
    proj_0,  # noqa: F401
)
