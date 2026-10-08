"""Public API for the link prediction package."""

from mage.link_prediction.constants import Activations  # noqa: F401
from mage.link_prediction.constants import Aggregators  # noqa: F401
from mage.link_prediction.constants import Context  # noqa: F401
from mage.link_prediction.constants import Devices  # noqa: F401
from mage.link_prediction.constants import Metrics  # noqa: F401
from mage.link_prediction.constants import Models  # noqa: F401
from mage.link_prediction.constants import Optimizers  # noqa: F401
from mage.link_prediction.constants import Parameters  # noqa: F401
from mage.link_prediction.constants import Predictors  # noqa: F401
from mage.link_prediction.constants import Reindex  # noqa: F401
from mage.link_prediction.link_prediction_util import add_self_loop  # noqa: F401
from mage.link_prediction.link_prediction_util import classify  # noqa: F401
from mage.link_prediction.link_prediction_util import inner_train  # noqa: F401
from mage.link_prediction.link_prediction_util import preprocess  # noqa: F401
from mage.link_prediction.link_prediction_util import proj_0  # noqa: F401
