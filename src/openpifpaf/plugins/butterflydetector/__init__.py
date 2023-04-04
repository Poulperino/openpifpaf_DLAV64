import openpifpaf
from .datasets.dataloader import UAVDTDataLoader, VisdroneDataLoader
from . import headmeta
from .heads import CompositeField
from .losses.loss import CompositeLoss
from .decoder import Butterfly
from .hrnet import HRNet

def register():
    openpifpaf.DATAMODULES['uavdt'] = UAVDTDataLoader
    openpifpaf.DATAMODULES['visdrone'] = VisdroneDataLoader

    openpifpaf.HEADS[headmeta.Butterfly] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly] = CompositeLoss
    openpifpaf.DECODERS.add(Butterfly)

    openpifpaf.HEADS[headmeta.Butterfly_LaplaceWH] = CompositeField
    openpifpaf.LOSSES[headmeta.Butterfly_LaplaceWH] = CompositeLoss

    openpifpaf.BASE_TYPES.add(HRNet)
    openpifpaf.BASE_FACTORIES['hrnetw32det'] = lambda: HRNet(cfg_file='w32_384x288_adam_lr1e-3.yaml', shortname='HRNet', detection=True, is_train=True)
    openpifpaf.BASE_FACTORIES['hrnetw48det'] = lambda: HRNet(cfg_file='w48_384x288_adam_lr1e-3.yaml', shortname='HRNet', detection=True, is_train=True)
