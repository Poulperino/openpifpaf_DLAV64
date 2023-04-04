import openpifpaf

from .dataloader.dataloader import Yolov7DataLoader
from .model.yolo import Model
from .butterflyencoder import headmeta
from ..butterflydetector import CompositeField, CompositeLoss
from .newDetector import headmeta as headmeta2
from .newDetector import Butterfly2Head, BF2Loss

def register():
    openpifpaf.DATAMODULES['yolov7data'] = Yolov7DataLoader

    openpifpaf.BASE_TYPES.add(Model)
    openpifpaf.BASE_FACTORIES['yolov7'] = lambda: Model(cfg='src/openpifpaf/plugins/yolov7/model/cfg/yolov7.yaml', ch=3)
    openpifpaf.BASE_FACTORIES['yolov7w6'] = lambda: Model(cfg='src/openpifpaf/plugins/yolov7/model/cfg/yolov7-w6.yaml', ch=3)

    openpifpaf.HEADS[headmeta.Butterfly] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly] = CompositeLoss

    openpifpaf.HEADS[headmeta2.Butterfly2] = Butterfly2Head
    openpifpaf.LOSSES[headmeta2.Butterfly2] = BF2Loss
