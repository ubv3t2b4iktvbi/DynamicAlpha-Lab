from .base import BaseForecastModel
from .rc import RCConfig, ReservoirTemplateFactory, PureRCModel
from .sindy import FullSINDyConfig, FullObservableSINDy, SlowBackboneSINDy, SlowSINDyConfig, SlowSINDyOnlyModel
from .hybrid import (
    ResidualLinearConfig,
    ResidualRCConfig,
    SlowSINDyDeltaLinearModel,
    SlowSINDyDeltaRCModel,
    SlowSINDyLevelLinearModel,
    SlowSINDyLevelRCModel,
)
from .ngrc import (
    NGRCConfig,
    RCNGRCConfig,
    ResidualNGRCConfig,
    ResidualRCNGRCConfig,
    PureNGRCModel,
    HybridRCNGRCModel,
    SlowSINDyDeltaNGRCModel,
    SlowSINDyDeltaHybridModel,
    quadratic_feature_dim,
)
