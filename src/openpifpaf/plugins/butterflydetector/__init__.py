import openpifpaf
from .datasets.dataloader import UAVDTDataLoader, VisdroneDataLoader
from . import headmeta
from .heads import CompositeField
from .losses.loss import CompositeLoss

def register():
    openpifpaf.DATAMODULES['uavdt'] = UAVDTDataLoader
    openpifpaf.DATAMODULES['visdrone'] = VisdroneDataLoader

    openpifpaf.HEADS[headmeta.Butterfly] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly] = CompositeLoss
