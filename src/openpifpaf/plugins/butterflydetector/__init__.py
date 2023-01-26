import openpifpaf
from .datasets.data_loader import UAVDTDataLoader
from . import headmeta
from .heads import CompositeField
from .losses.loss import CompositeLoss

def register():
    openpifpaf.DATAMODULES['uavdt'] = UAVDTDataLoader

    openpifpaf.HEADS[headmeta.Butterfly] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly] = CompositeLoss
