import argparse
import openpifpaf
import torchvision
import torch
import mmcv
from mmdet.apis import init_detector
from mmdet.models import build_backbone, build_neck

class MMDet_Models(openpifpaf.network.BaseNetwork):
    stride = 4
    config_list = {
        'resnet50_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/r50_fpn.py',
        'resnet101_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/r101_fpn.py',
        'hrnetw32v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet32v2_fpn.py',
        'hrnetw48v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet48v2_fpn.py',
        'hrnetw40v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet40v2_fpn.py',
        'hrnetw48v2': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet48v2.py',
    }
    def __init__(self, name):
        super(MMDet_Models, self).__init__(name, stride=self.stride, out_features=256)
        config = mmcv.Config.fromfile(self.config_list[name])['model']
        self.backbone = build_backbone(config.get('backbone'))
        self.neck_det, self.neck_rel = None, None
        if config.get('neck_det', False):
            self.neck_det = build_neck(config.get('neck_det'))
        if config.get('neck_rel', False):
            self.neck_rel = build_neck(config.get('neck_rel'))


    def forward(self, x):
        x = self.backbone(x)
        if self.neck_det is None:
            return x
        elif self.neck_rel is None:
            det_feats = self.neck_det(x)
            return det_feats[0]
        else:
            det_feats = self.neck_det(x)
            rel_feats = self.neck_rel(x)
            return [det_feats, rel_feats]

def register():
    openpifpaf.BASE_TYPES.add(MMDet_Models)
    openpifpaf.BASE_FACTORIES['mmdet_r50_fpn'] = lambda: MMDet_Models(name='resnet50_fpn')
    openpifpaf.BASE_FACTORIES['mmdet_r101_fpn'] = lambda: MMDet_Models(name='resnet101_fpn')
    openpifpaf.BASE_FACTORIES['mmdet_hrw48v2'] = lambda: MMDet_Models(name='hrnetw48v2')
    openpifpaf.BASE_FACTORIES['mmdet_hrw48v2_fpn'] = lambda: MMDet_Models(name='hrnetw48v2_fpn')
    openpifpaf.BASE_FACTORIES['mmdet_hrw32v2_fpn'] = lambda: MMDet_Models(name='hrnetw32v2_fpn')
    openpifpaf.BASE_FACTORIES['mmdet_hrw40v2_fpn'] = lambda: MMDet_Models(name='hrnetw40v2_fpn')
