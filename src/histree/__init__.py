from .gb.model import HistogramGradientBoostingRegressor
from .rf.model import HistogramRandomForestRegressor
from .swf.model import HistogramSequentialWeightedForestRegressor

__all__ = [
    "HistogramRandomForestRegressor",
    "HistogramGradientBoostingRegressor",
    "HistogramSequentialWeightedForestRegressor",
]

__version__ = "0.1.0"
