import argparse
import openpifpaf
import torchvision
import torch
import torch.nn.functional as F
import mmcv
from mmdet.models import build_backbone, build_neck, NECKS
# from mmdet.models.registry import NECKS
from mmcv.cnn import ConvModule


@NECKS.register_module()
class DetectionNeck(torch.nn.Module):

    """DetectionNeck

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')):

        super(DetectionNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        # self.fpn_convs = nn.ModuleList()
        # for i in range(self.num_outs):
        #     self.fpn_convs.append(
        #         ConvModule(
        #             out_channels,
        #             out_channels,
        #             kernel_size=3,
        #             padding=1,
        #             stride=stride,
        #             conv_cfg=self.conv_cfg,
        #             act_cfg=None))
        #
        # if pooling_type == 'MAX':
        #     self.pooling = F.max_pool2d
        # else:
        #     self.pooling = F.avg_pool2d

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        # outs = [out]
        # for i in range(1, self.num_outs):
        #     outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        # outputs = []

        # for i in range(self.num_outs):
        #     if outs[i].requires_grad and self.with_cp:
        #         tmp_out = checkpoint(self.fpn_convs[i], outs[i])
        #     else:
        #         tmp_out = self.fpn_convs[i](outs[i])
        #     outputs.append(tmp_out)
        return [out]

class MMDet_Models(openpifpaf.network.BaseNetwork):
    stride = 4
    config_list = {
        'resnet50_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/r50_fpn.py',
        'resnet101_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/r101_fpn.py',
        'hrnetw32v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet32v2_fpn.py',
        'hrnetw48v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet48v2_fpn.py',
        'hrnetw40v2_fpn': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet40v2_fpn.py',
        'hrnetw48v2': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet48v2.py',
        'hrnetw32v2det': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet32v2det.py',
        'hrnetw48v2det': 'src/openpifpaf/plugins/mmdet_models/configs/hrnet48v2det.py',
    }
    def __init__(self, name):
        config = mmcv.Config.fromfile(self.config_list[name])['model']
        super(MMDet_Models, self).__init__(name, stride=self.stride, out_features=config.get('neck_det')['out_channels'])

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
    openpifpaf.BASE_FACTORIES['mmdet_hrw32v2det'] = lambda: MMDet_Models(name='hrnetw32v2det')
    openpifpaf.BASE_FACTORIES['mmdet_hrw48v2det'] = lambda: MMDet_Models(name='hrnetw48v2det')
    openpifpaf.BASE_FACTORIES['mmdet_hrw32v2_fpn'] = lambda: MMDet_Models(name='hrnetw32v2_fpn')
    openpifpaf.BASE_FACTORIES['mmdet_hrw40v2_fpn'] = lambda: MMDet_Models(name='hrnetw40v2_fpn')
