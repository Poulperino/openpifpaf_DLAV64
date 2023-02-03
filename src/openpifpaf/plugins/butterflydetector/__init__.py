import openpifpaf
from .datasets.dataloader import UAVDTDataLoader, VisdroneDataLoader
from . import headmeta
from .heads import CompositeField
from .losses.loss import CompositeLoss
from .decoder import Butterfly
def register():
    openpifpaf.DATAMODULES['uavdt'] = UAVDTDataLoader
    openpifpaf.DATAMODULES['visdrone'] = VisdroneDataLoader

    openpifpaf.HEADS[headmeta.Butterfly] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly] = CompositeLoss
    openpifpaf.DECODERS.add(Butterfly)
