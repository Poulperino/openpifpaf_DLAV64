from . import tracking_heads
from .nets import model_defaults
from .tracking_base import TrackingBase
from ..signal import Signal

MODEL_MIGRATION = set()

import torch
def model_migration2(net_cpu):
    model_defaults(net_cpu)

    for m in net_cpu.modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if not hasattr(m, 'padding_mode'):  # introduced in PyTorch 1.1.0
            m.padding_mode = 'zeros'

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    if not hasattr(net_cpu, 'head_strides'):
        net_cpu.head_strides = [
            net_cpu.base_net.stride // (2 ** getattr(h, '_quad', 0))
            for h in net_cpu.head_nets
        ]

    if not hasattr(net_cpu, 'head_names'):
        net_cpu.head_names = [
            h.meta.name for h in net_cpu.head_nets
        ]

    for head in net_cpu.head_nets:
        if not hasattr(head, 'dropout') or head.dropout is None:
            head.dropout = torch.nn.Dropout2d(p=0.0)
        if not hasattr(head, '_quad'):
            if hasattr(head, 'quad'):
                head._quad = head.quad  # pylint: disable=protected-access
            else:
                head._quad = 0  # pylint: disable=protected-access
        if not hasattr(head, 'scale_conv'):
            head.scale_conv = None
        if not hasattr(head, 'reg1_spread'):
            head.reg1_spread = None
        if not hasattr(head, 'reg2_spread'):
            head.reg2_spread = None
        if head.meta.name == 'pif17' and getattr(head, 'scale_conv') is not None:
            head.name = 'pifs17'
        if head._quad == 1 and not hasattr(head, 'dequad_op'):  # pylint: disable=protected-access
            head.dequad_op = torch.nn.PixelShuffle(2)
        if not hasattr(head, 'class_convs') and hasattr(head, 'class_conv'):
            head.class_convs = torch.nn.ModuleList([head.class_conv])



# pylint: disable=protected-access
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for m in net_cpu.modules():
        if not hasattr(m, '_non_persistent_buffers_set'):
            m._non_persistent_buffers_set = set()

    for m in net_cpu.modules():
        if m.__class__.__name__ != 'InvertedResidualK':
            continue
        if not hasattr(m, 'branch1'):
            m.branch1 = None

    if not hasattr(net_cpu, 'head_nets') and hasattr(net_cpu, '_head_nets'):
        net_cpu.head_nets = net_cpu._head_nets

    for hn_i, hn in enumerate(net_cpu.head_nets):
        if not hn.meta.base_stride:
            hn.meta.base_stride = net_cpu.base_net.stride
        if hn.meta.head_index is None:
            hn.meta.head_index = hn_i
        if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
            hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)

    for mm in MODEL_MIGRATION:
        mm(net_cpu)


def fix_feature_cache(model):
    for m in model.modules():
        if not isinstance(m, TrackingBase):
            continue
        m.reset()


def subscribe_cache_reset(model):
    for m in model.modules():
        if not isinstance(m, TrackingBase):
            continue
        Signal.subscribe('eval_reset', m.reset)


def tcaf_shared_preprocessing(model):
    for m in model.modules():
        if not isinstance(m, tracking_heads.Tcaf):
            continue

        # pylint: disable=protected-access
        tracking_heads.Tcaf._global_feature_reduction = m.feature_reduction
        tracking_heads.Tcaf._global_feature_compute = m.feature_compute
        return


MODEL_MIGRATION.add(fix_feature_cache)
MODEL_MIGRATION.add(subscribe_cache_reset)
MODEL_MIGRATION.add(tcaf_shared_preprocessing)
