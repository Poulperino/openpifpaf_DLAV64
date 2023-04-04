import functools
import logging
import torch
import numpy as np
from .utils import *
import argparse
from ..utils import index_field

LOG = logging.getLogger(__name__)

class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    multiplicity_correction = False
    independence_scale = 3.0
    reg_loss_name = 'laplace'
    # def __init__(self, head_name, regression_loss, *,
    #              n_vectors, n_scales, sigmas=None, margin=False, iou_loss=0, focal_loss=False):
    def __init__(self, head_meta):
        super(CompositeLoss, self).__init__()

        iou_loss = 0
        if self.reg_loss_name == 'smoothl1':
            regression_loss = SmoothL1Loss(r_smooth)
        elif self.reg_loss_name == 'l1':
            regression_loss = l1_loss
        elif self.reg_loss_name == 'laplace' or self.reg_loss_name == 'laplace_focal':
            regression_loss = laplace_loss
        elif self.reg_loss_name == 'laplace_iou':
            regression_loss = laplace_loss
            iou_loss = 1
        elif self.reg_loss_name == 'laplace_siou':
            regression_loss = laplace_loss
            iou_loss = 2
        elif self.reg_loss_name == 'iou_only':
            regression_loss = laplace_loss
            iou_loss = 3
        elif self.reg_loss_name == 'siou_only':
            regression_loss = laplace_loss
            iou_loss = 4
        elif self.reg_loss_name is None:
            regression_loss = laplace_loss
        else:
            raise Exception('unknown regression loss type {}'.format(reg_loss_name))

        focal_loss='focal' in self.reg_loss_name
        n_vectors = head_meta.n_vectors
        n_scales = head_meta.n_scales
        head_name = head_meta.name
        sigmas = None
        margin = False

        self.focal_loss = focal_loss
        if self.focal_loss:
            self.focal = FocalLoss(alpha=0.25, gamma=2, eps=1e-7)
        self.n_vectors = n_vectors
        self.n_scales = n_scales
        self.iou_loss = iou_loss

        if not ('butterfly' in head_name or 'repulse' in head_name):
            self.scales_butterfly = None
        if self.n_scales and not ('butterfly' in head_name or 'repulse' in head_name):
            assert len(sigmas) == n_scales
        elif self.n_scales:
            scales_butterfly = [[1.0] for _ in range(2)]
            scales_butterfly = torch.tensor(scales_butterfly)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            scales_butterfly = torch.unsqueeze(scales_butterfly, -1)
            self.register_buffer('scales_butterfly', scales_butterfly)
        else:
            self.scales_butterfly = None

        if sigmas is None:
            sigmas = [[1.0] for _ in range(n_vectors)]
        if sigmas is not None and 'butterfly' in head_name:
            assert len(sigmas) == n_vectors
            scales_to_kp = torch.tensor(sigmas)
            scales_to_kp = torch.unsqueeze(scales_to_kp, 0)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            self.register_buffer('scales_to_kp', scales_to_kp)
        else:
            self.scales_to_kp = None

        self.regression_loss = regression_loss or laplace_loss

        if self.iou_loss>0:
            self.field_names = (
                ['{}.c'.format(head_name)] +
                ['{}.vec{}'.format(head_name, i + 1) for i in range(self.n_vectors)] +
                #['{}.scales{}'.format(head_name, i + 1) for i in range(self.n_scales)] +
                ['{}.iou{}'.format(head_name, i + 1) for i in range(self.n_vectors)]
            )
        else:
            self.field_names = (
                ['{}.c'.format(head_name)] +
                ['{}.vec{}'.format(head_name, i + 1) for i in range(self.n_vectors)] +
                ['{}.scales{}'.format(head_name, i + 1) for i in range(self.n_scales)]
            )
        self.margin = margin
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

        self.repulsion_loss = None
        if 'repulse' in head_name:
            self.repulsion_loss = RepulsionLoss()
        LOG.debug('%s: n_vectors = %d, n_scales = %d, len(sigmas) = %d, margin = %s',
                  head_name, n_vectors, n_scales, len(sigmas), margin)

    def cli(parser):
        group = parser.add_argument_group('Butterfly losses')
        # pass
        # group.add_argument('--lambdas', default=[30.0, 2.0, 2.0, 50.0, 3.0, 3.0],
        #                    type=float, nargs='+',
        #                    help='prefactor for head losses')
        # group.add_argument('--r-smooth', type=float, default=0.0,
        #                    help='r_{smooth} for SmoothL1 regressions')
        group.add_argument('--bf-regression-loss', default='laplace',
                           choices=['smoothl1', 'smootherl1', 'l1', 'laplace', 'laplace_iou', 'laplace_siou', 'iou_only', 'siou_only', 'laplace_focal'],
                           help='type of regression loss')
        # group.add_argument('--background-weight', default=1.0, type=float,
        #                    help='[experimental] BCE weight of background')
        # group.add_argument('--margin-loss', default=False, action='store_true',
        #                    help='[experimental]')
        # group.add_argument('--auto-tune-mtl', default=False, action='store_true',
        #                    help='[experimental]')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.reg_loss_name = args.bf_regression_loss

    def forward(self, *args):  # pylint: disable=too-many-statements
        LOG.debug('loss for %s', self.field_names)

        x, t = args

        assert len(x) == 1 + 2 * self.n_vectors + self.n_scales
        x_intensity = x[0]
        x_regs = x[1:1 + self.n_vectors]
        x_spreads = x[1 + self.n_vectors:1 + 2 * self.n_vectors]
        x_scales = []
        if self.n_scales:
            x_scales = x[1 + 2 * self.n_vectors:1 + 2 * self.n_vectors + self.n_scales]

        #assert len(t) == 1 + self.n_vectors + self.n_scales

        ## Original Butterfly
        # target_intensity = t[0]
        # target_regs = t[1:1 + self.n_vectors]
        # target_scales = t[1 + self.n_vectors:1 + self.n_vectors + self.n_scales]
        target_intensity = t[:, :, 0]
        target_regs = [t[:, :, 1+2*indx_vec:1+2*indx_vec + 2] for indx_vec in range(self.n_vectors)]
        target_scales = t[:, :, 1 + 2*self.n_vectors:1 + 2*self.n_vectors + self.n_scales].permute(2,0,1,3,4)

        # bce_masks = (target_intensity[:, :-1] + target_intensity[:, -1:]) > 0.5
        bce_masks = torch.isnan(target_intensity).bitwise_not_()
        if not torch.any(bce_masks):
            return None, None, None

        batch_size = x_intensity.shape[0]
        LOG.debug('batch size = %d', batch_size)

        bce_x_intensity = x_intensity

        ## Original Butterfly
        # bce_target_intensity = target_intensity[:, :-1]
        bce_target_intensity = target_intensity
        if self.bce_blackout:
            bce_x_intensity = bce_x_intensity[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            bce_target_intensity = bce_target_intensity[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_intensity.shape, bce_target_intensity.shape, bce_masks.shape)
        # bce_masks = (
        #     bce_masks
        #     & ((x_intensity > -4.0) | ((x_intensity < -4.0) & (target_intensity[:, :-1] == 1)))
        #     & ((x_intensity < 4.0) | ((x_intensity > 4.0) & (target_intensity[:, :-1] == 0)))
        # )
        bce_target = torch.masked_select(bce_target_intensity, bce_masks)
        bce_weight = None
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target)
            bce_weight[bce_target == 0] = self.background_weight

        if self.focal_loss:
            ce_loss = self.focal(torch.masked_select(bce_x_intensity, bce_masks), bce_target, bce_weight)#/batch_size
        else:
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.masked_select(bce_x_intensity, bce_masks),
                bce_target,
                reduction='sum',
            ) / 100.0 / batch_size
        reg_losses = [None for _ in target_regs]
        #reg_masks = target_intensity[:, :-1] > 0.5
        reg_masks = target_intensity > 0.5
        if torch.any(reg_masks):
            weight = None
            if self.multiplicity_correction:
                assert len(target_regs) == 2
                lengths = torch.norm(target_regs[0] - target_regs[1], dim=2)
                multiplicity = (lengths - 3.0) / self.independence_scale
                multiplicity = torch.clamp(multiplicity, min=1.0)
                multiplicity = torch.masked_select(multiplicity, reg_masks)
                weight = 1.0 / multiplicity

            reg_losses = []
            if self.iou_loss<3:
                for i, (x_reg, x_spread, target_reg) in enumerate(zip(x_regs, x_spreads, target_regs)):
                    if hasattr(self.regression_loss, 'scale'):
                        assert self.scales_to_kp is not None
                        self.regression_loss.scale = torch.masked_select(
                            torch.clamp(target_scales[i], 0.1, 1000.0),  # pylint: disable=unsubscriptable-object
                            reg_masks,
                        )

                    reg_losses.append(self.regression_loss(
                        torch.masked_select(x_reg[:, :, 0], reg_masks),
                        torch.masked_select(x_reg[:, :, 1], reg_masks),
                        torch.masked_select(x_spread, reg_masks),
                        torch.masked_select(target_reg[:, :, 0], reg_masks),
                        torch.masked_select(target_reg[:, :, 1], reg_masks),
                        weight=(weight if weight is not None else 1.0) * 0.1,
                    ) / 100.0 / batch_size)

        scale_losses = []
        wh_losses = []
        if self.iou_loss==0:
            if not self.scales_butterfly is None:
                if x_scales:
                    wh_losses = [
                        torch.nn.functional.l1_loss(
                            torch.masked_select(x_scale, torch.isnan(target_wh) == 0),
                            torch.masked_select(target_wh, torch.isnan(target_wh) == 0),
                            reduction='sum',
                        ) / 1000.0 / batch_size
                        for x_scale, target_wh, scale_to_kp in zip(x_scales, target_scales, self.scales_butterfly)
                    ]
            else:
                if x_scales:
                    scale_losses = [
                        torch.nn.functional.l1_loss(
                            torch.masked_select(x_scale, torch.isnan(target_scale) == 0),
                            torch.masked_select(target_scale, torch.isnan(target_scale) == 0),
                            reduction='sum',
                        ) / 1000.0 / batch_size
                        for x_scale, target_scale, scale_to_kp in zip(x_scales, target_scales, self.scales_to_kp)
                    ]

        iou_losses = []
        repgt_losses = []
        repbbox_losses = []
        if self.iou_loss>0 or self.repulsion_loss:
            if self.repulsion_loss:
                fields_ids = t[-1]
            for i, (x_reg, target_reg) in enumerate(zip(x_regs, target_regs)):
                index_fields = index_field(x_reg[:, :, 0:2].shape[-2:])
                index_fields = np.expand_dims(index_fields, 0)
                index_fields = np.expand_dims(index_fields, 0)
                joint_fields_pred = torch.from_numpy(index_fields.copy()).cuda() + x_reg[:, :, 0:2]

                joint_fields_gt = torch.from_numpy(index_fields.copy()).cuda() + target_reg[:, :, 0:2]


                w_pred = torch.exp(torch.masked_select(x_scales[0], reg_masks))
                h_pred = torch.exp(torch.masked_select(x_scales[1], reg_masks))
                x_pred = torch.masked_select(joint_fields_pred[:, :, 0], reg_masks)
                y_pred = torch.masked_select(joint_fields_pred[:, :, 1], reg_masks)


                w_gt = torch.exp(torch.masked_select(target_scales[0], reg_masks))
                h_gt = torch.exp(torch.masked_select(target_scales[1], reg_masks))
                x_gt = torch.masked_select(joint_fields_gt[:,:,0], reg_masks)
                y_gt = torch.masked_select(joint_fields_gt[:,:,1], reg_masks)
                if self.repulsion_loss:
                    x_gt = x_gt - w_gt/2
                    y_gt = y_gt - h_gt/2
                    x_pred = x_pred - w_pred/2
                    y_pred = y_pred - h_pred/2
                    bbox_pred = torch.cat((torch.unsqueeze(x_pred,1), torch.unsqueeze(y_pred,1), torch.unsqueeze(w_pred,1), torch.unsqueeze(h_pred,1)), 1)
                    bbox_gt = torch.cat((torch.unsqueeze(x_gt,1), torch.unsqueeze(y_gt,1), torch.unsqueeze(w_gt,1), torch.unsqueeze(h_gt,1)), 1)
                    repgt_loss, repbbox_loss = self.repulsion_loss(bbox_pred, bbox_gt, ids = torch.masked_select(fields_ids, reg_masks))
                    repgt_losses.append(repgt_loss)
                    repbbox_losses.append(repbbox_loss)

                else:
                    if self.iou_loss%2 != 0:
                        iou_pred = ratio_iou_scripted(x_pred, y_pred, \
                                                      w_pred, h_pred, \
                                                      x_gt, y_gt, \
                                                      w_gt, h_gt)
                        iou_loss = torch.nn.functional.binary_cross_entropy(
                            iou_pred,
                            torch.ones_like(iou_pred),
                            reduction='sum',
                        )/ 1000.0/ batch_size
                    elif self.iou_loss != 0:
                        iou_pred = ratio_siou_scripted(x_pred, y_pred, \
                                                      w_pred, h_pred, \
                                                      x_gt, y_gt, \
                                                      w_gt, h_gt)

                        iou_loss = (1 - iou_pred).sum()/1000.0/ batch_size
                    iou_losses.append(iou_loss)


        margin_losses = [None for _ in target_regs] if self.margin else []
        if self.margin and torch.any(reg_masks):
            margin_losses = []
            for i, (x_reg, target_reg) in enumerate(zip(x_regs, target_regs)):
                margin_losses.append(quadrant_margin_loss(
                    torch.masked_select(x_reg[:, :, 0], reg_masks),
                    torch.masked_select(x_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 0], reg_masks),
                    torch.masked_select(target_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 2], reg_masks),
                    torch.masked_select(target_reg[:, :, 3], reg_masks),
                    torch.masked_select(target_reg[:, :, 4], reg_masks),
                    torch.masked_select(target_reg[:, :, 5], reg_masks),
                ) / 100.0 / batch_size)

        return [ce_loss] + reg_losses + scale_losses + margin_losses + wh_losses + iou_losses + repgt_losses + repbbox_losses

# def factory_from_args(args):
#     # apply for CompositeLoss
#     CompositeLoss.background_weight = args.background_weight
#
#     return factory(
#         args.headnets,
#         args.lambdas,
#         reg_loss_name=args.regression_loss,
#         r_smooth=args.r_smooth,
#         device=args.device,
#         margin=args.margin_loss,
#         auto_tune_mtl=args.auto_tune_mtl,
#     )


# def loss_parameters(head_name):
#     n_vectors = 1
#
#     n_scales = 2
#     return {
#         'n_vectors': n_vectors,
#         'n_scales': n_scales,
#     }
#
#
# def factory(head_names, lambdas, *,
#             reg_loss_name=None, r_smooth=None, device=None, margin=False,
#             auto_tune_mtl=False):
#     if isinstance(head_names[0], (list, tuple)):
#         return [factory(hn, lam,
#                         reg_loss_name=reg_loss_name,
#                         r_smooth=r_smooth,
#                         device=device,
#                         margin=margin)
#                 for hn, lam in zip(head_names, lambdas)]
#
#     head_names = [h for h in head_names if h not in ('skeleton', 'tskeleton')]
#     iou_loss = 0
#     if reg_loss_name == 'smoothl1':
#         reg_loss = SmoothL1Loss(r_smooth)
#     elif reg_loss_name == 'l1':
#         reg_loss = l1_loss
#     elif reg_loss_name == 'laplace' or reg_loss_name == 'laplace_focal':
#         reg_loss = laplace_loss
#     elif reg_loss_name == 'laplace_iou':
#         reg_loss = laplace_loss
#         iou_loss = 1
#     elif reg_loss_name == 'laplace_siou':
#         reg_loss = laplace_loss
#         iou_loss = 2
#     elif reg_loss_name == 'iou_only':
#         reg_loss = laplace_loss
#         iou_loss = 3
#     elif reg_loss_name == 'siou_only':
#         reg_loss = laplace_loss
#         iou_loss = 4
#     elif reg_loss_name is None:
#         reg_loss = laplace_loss
#     else:
#         raise Exception('unknown regression loss type {}'.format(reg_loss_name))
#
#     losses = [CompositeLoss(head_name, reg_loss,
#                             margin=margin, iou_loss=iou_loss, focal_loss='focal' in reg_loss_name, **loss_parameters(head_name))
#               for head_name in head_names]
#     if auto_tune_mtl:
#         loss = MultiHeadLossAutoTune(losses, lambdas)
#     else:
#         loss = MultiHeadLoss(losses, lambdas)
#
#     if device is not None:
#         loss = loss.to(device)
#
#     return loss
