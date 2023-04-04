import argparse
import logging

import torch
import numpy as np
import math

# from . import components

LOG = logging.getLogger(__name__)

def index_field(shape):
    yx = np.indices(shape, dtype=np.float32)
    xy = np.flip(yx, axis=0)
    return xy

class ComboIOUL1(torch.nn.Module):
    lambdas = [1.0, 1.0]
    def __init__(self, lambdas=[1.0, 1.0]):
        super(ComboIOUL1, self).__init__()
        self.lambdas = lambdas

    def _ciou(self, bboxes1, bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        cious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return cious
        w1 = bboxes1[:, 2]
        h1 = bboxes1[:, 3]
        w2 = bboxes2[:, 2]
        h2 = bboxes2[:, 3]
        area1 = w1 * h1
        area2 = w2 * h2
        center_x1 = bboxes1[:, 0]
        center_y1 = bboxes1[:, 1]
        center_x2 = bboxes2[:, 0]
        center_y2 = bboxes2[:, 1]

        inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
        inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
        inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
        inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
        inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

        c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
        c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
        c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
        c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
        c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

        union = area1+area2-inter_area
        u = (inter_diag) / c_diag
        iou = inter_area / union
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = (iou>0.5).float()
            alpha= S*v/(1-iou+v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious,min=-1.0,max = 1.0)
        return cious

    def __call__(self, x_regs, t_regs):
        # x_regs = x_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)
        # t_regs = t_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)

        d = x_regs - t_regs
        d = torch.linalg.norm(d, ord=2, dim=1, keepdim=True)

        ciou_loss = (1 - self._ciou(x_regs, t_regs))
        ciou_loss = ciou_loss #/ (t_regs.shape[0] + 1e-4)
        return self.lambdas[0]*d + self.lambdas[1]*ciou_loss.unsqueeze(-1)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.m = torch.nn.Sigmoid()
        self.previous=None

    def forward(self, preds, gt):
        pos_mask = gt.long().eq(1)
        neg_mask = gt.long().lt(1)
        res = self.m(preds)

        pos_loss = self.alpha*torch.log(res[pos_mask]) * torch.pow(1 - res[pos_mask], self.gamma)
        neg_loss = (1-self.alpha)*torch.log(1 - res[neg_mask]) * torch.pow(res[neg_mask], self.gamma)#*neg_weights
        loss = torch.cat((pos_loss, neg_loss), 0)

        return - loss.sum()/max((gt==1).sum(),1.0)

class SmoothL1Loss(object):
    def __init__(self):
        self.smoothloss = torch.nn.SmoothL1Loss(reduction='none')

    def __call__(self, x_regs, t_regs, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        x_regs = x_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)
        t_regs = t_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)

        loesses = self.smoothloss(x_regs, t_regs)

        if weight is not None:
            losses = losses * weight
        return torch.sum(losses)

class L1Loss(object):
    def __init__(self):
        self.l1Loss = torch.nn.L1Loss(reduction='none')

    def __call__(self, x_regs, t_regs, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        x_regs = x_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)
        t_regs = t_regs.reshape(t_regs.shape[0], t_regs.shape[1] // 2, 2)

        losses = self.l1Loss(x_regs, t_regs)

        if weight is not None:
            losses = losses * weight
        return torch.sum(losses)

class BF2Loss(torch.nn.Module):

    soft_clamp_value = None #5.0
    IOUL1_lambdas = [1.0, 1.0]
    use_bce = False
    smoothL1 = False
    focal_loss = True

    def __init__(self, head_meta):
        super().__init__()
        self.n_confidences = head_meta.n_confidences
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_meta.name, self.n_vectors, self.n_scales)

        # if self.regression_loss == 'iou_l1':
        #     self.field_names = (
        #         '{}.{}.c'.format(head_meta.dataset, head_meta.name),
        #         '{}.{}.vec'.format(head_meta.dataset, head_meta.name),
        #         '{}.{}.iou'.format(head_meta.dataset, head_meta.name),
        #         '{}.{}.scales'.format(head_meta.dataset, head_meta.name),
        #     )
        # else:
        self.field_names = (
            '{}.{}.c'.format(head_meta.dataset, head_meta.name),
            '{}.{}.vec'.format(head_meta.dataset, head_meta.name),
            '{}.{}.scales'.format(head_meta.dataset, head_meta.name),
        )


        # self.bce_loss = components.Bce()
        if self.focal_loss:
            self.bce_loss = FocalLoss()
        else:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        if self.regression_loss == 'iou_l1':
            self.reg_loss = ComboIOUL1(self.IOUL1_lambdas)
        elif self.regression_loss == 'smoothl1':
            self.reg_loss = SmoothL1Loss()
        elif self.regression_loss == 'l1':
            self.reg_loss = L1Loss()

        self.scale_loss = L1Loss()

        # self.soft_clamp = None
        # if self.soft_clamp_value:
        #     self.soft_clamp = components.SoftClamp(self.soft_clamp_value)

        self.weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            self.weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)

        LOG.debug("The weights for the keypoints are %s", self.weights)
        # self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss')
        # group.add_argument('--soft-clamp', default=cls.soft_clamp_value, type=float,
        #                    help='soft clamp')
        group.add_argument('--bf2-IOUL1-lambdas', default=cls.IOUL1_lambdas, nargs=2, type=float,
                           help='weight of L1 and IOU loss')
        group.add_argument('--bf2-regression-loss', default='l1',
                           choices=['smoothl1', 'l1', 'iou_l1'],
                           help='type of regression loss')
        group.add_argument('--bf2-confidence-loss', default='focal',
                           choices=['focal', 'bce'],
                           help='type of confidence loss')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # cls.soft_clamp_value = args.soft_clamp
        cls.combo_IOUL1_lambdas = args.bf2_IOUL1_lambdas
        cls.regression_loss = args.bf2_regression_loss

        if args.bf2_confidence_loss == 'bce':
            cls.focal_loss = False
        elif args.bf2_confidence_loss == 'focal':
            cls.focal_loss = True
        # cls.use_bce = args.use_bce

    # pylint: disable=too-many-statements
    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)
        batch_size = t.shape[0]
        if t is None:
            return [None, None, None]
        assert x.shape[2] == self.n_confidences + self.n_vectors * 2 + self.n_scales
        assert t.shape[2] == self.n_confidences + self.n_vectors * 2 + self.n_scales

        # determine foreground and background masks based on ground truth
        t = torch.transpose(t, 2, 4)
        finite = torch.isfinite(t)
        t_confidence_raw = t[:, :, :, :, 0:self.n_confidences]
        bg_mask = torch.all(t_confidence_raw == 0.0, dim=4)
        c_mask = torch.all(t_confidence_raw > 0.0, dim=4)
        reg_mask = torch.all(finite[:, :, :, :, self.n_confidences:self.n_confidences + self.n_vectors * 2], dim=4)
        scale_mask = torch.all(finite[:, :, :, :, self.n_confidences + self.n_vectors * 2:], dim=4)

        # extract masked ground truth
        t_confidence_bg = t[bg_mask][:, 0:self.n_confidences]
        t_confidence = t[c_mask][:, 0:self.n_confidences]
        t_regs = t[reg_mask][:, self.n_confidences:self.n_confidences + self.n_vectors * 2]
        # t_sigma_min = t[reg_mask][
        #     :,
        #     self.n_confidences + self.n_vectors * 2:self.n_confidences + self.n_vectors * 3
        # ]
        # t_scales_reg = t[reg_mask][:, self.n_confidences + self.n_vectors * 2:]
        t_scales = t[scale_mask][:, self.n_confidences + self.n_vectors * 2:]

        # extract masked predictions
        x = torch.transpose(x, 2, 4)
        x_confidence_bg = x[bg_mask][:, 0:self.n_confidences]
        # x_logs2_c = x[c_mask][:, 0:1]
        x_confidence = x[c_mask][:, 0:self.n_confidences]
        # x_logs2_reg = x[reg_mask][:, 0:1]
        x_regs = x[reg_mask][:, self.n_confidences:self.n_confidences + self.n_vectors * 2]
        # x_logs2_scale = x[scale_mask][:, 0:1]
        # x_scales_reg = x[reg_mask][:, self.n_confidences + self.n_vectors * 2:]
        x_scales = x[scale_mask][:, self.n_confidences + self.n_vectors * 2:]

        # impute t_scales_reg with predicted values
        # t_scales_reg = t_scales_reg.clone()
        # invalid_t_scales_reg = torch.isnan(t_scales_reg)
        # t_scales_reg[invalid_t_scales_reg] = \
        #     torch.nn.functional.softplus(x_scales_reg.detach()[invalid_t_scales_reg])
        # import pdb; pdb.set_trace()
        if self.focal_loss:
            l_ce = self.bce_loss(x[:,:,:,:, 0:self.n_confidences], t[:,:,:,:, 0:self.n_confidences])
        else:
            l_confidence_bg = self.bce_loss(x_confidence_bg, t_confidence_bg)
            l_confidence = self.bce_loss(x_confidence, t_confidence)
            l_ce = 0.1*torch.sum(l_confidence_bg) + torch.sum(l_confidence)
            l_ce = l_ce/t_confidence.shape[0]

        if self.regression_loss == 'iou_l1':
            index_fields = index_field(x.shape[2:4])
            index_fields = torch.from_numpy(index_fields.copy()).cuda()
            index_fields = torch.permute(index_fields, (1, 2, 0))
            index_fields = index_fields.expand(*x.shape[:2],-1,-1,-1)
            index_fields = index_fields[reg_mask]
            x_regs[:, 0:2] = index_fields + x_regs[:, 0:2]
            t_regs[:, 0:2] = index_fields + t_regs[:, 0:2]
            # l_reg = self.reg_loss(x_regs, t_regs)
            # l_reg = l_reg_l1 + l_reg_iou
        # l_reg = self.reg_loss(x_regs, t_regs, t_sigma_min, t_scales_reg)
        # l_scale = self.scale_loss(x_scales, t_scales)
        # else:
        l_reg = self.reg_loss(x_regs, t_regs)
        l_reg = torch.sum(l_reg) / reg_mask.sum()
        l_scale = self.scale_loss(x_scales, t_scales)
        l_scale = torch.sum(l_scale) / scale_mask.sum()

        # # softclamp
        # if self.soft_clamp is not None:
        #     l_confidence_bg = self.soft_clamp(l_confidence_bg)
        #     l_confidence = self.soft_clamp(l_confidence)
        #     l_reg = self.soft_clamp(l_reg)
        #     l_scale = self.soft_clamp(l_scale)

        # --- composite uncertainty
        # # c
        # if not self.use_bce:
        #     x_logs2_c = 3.0 * torch.tanh(x_logs2_c / 3.0)
        #     l_confidence = 0.5 * l_confidence * torch.exp(-x_logs2_c) + 0.5 * x_logs2_c

        # reg
        # x_logs2_reg = 3.0 * torch.tanh(x_logs2_reg / 3.0)
        # We want sigma = b*0.5. Therefore, log_b = 0.5 * log_s2 + log2
        # x_logb = 0.5 * x_logs2_reg + 0.69314
        # reg_factor = torch.exp(-x_logb)
        # x_logb = x_logb.unsqueeze(1)
        # reg_factor = reg_factor.unsqueeze(1)
        # if self.n_vectors > 1:
            # x_logb = torch.repeat_interleave(x_logb, self.n_vectors, 1)
            # reg_factor = torch.repeat_interleave(reg_factor, self.n_vectors, 1)
        # if not self.combo_IOUL1:
        #     l_reg = l_reg * reg_factor + x_logb
        # scale
        # scale_factor = torch.exp(-x_logs2)
        # for i in range(self.n_scales):
        #     l_scale_component = l_scale[:, i]
        #     l_scale_component = l_scale_component * scale_factor + 0.5 * x_logs2

        # if self.weights is not None:
        #     full_weights = torch.empty_like(t_confidence_raw)
        #     full_weights[:] = self.weights
        #     l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
        #     l_confidence = full_weights[c_mask] * l_confidence
        #     l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
        #     l_scale = full_weights[scale_mask] * l_scale

        # if self.regression_loss == 'iou_l1':
        #     losses = [
        #         l_ce,
        #         torch.sum(l_reg_l1) / reg_mask.sum(),
        #         torch.sum(l_reg_iou) / reg_mask.sum(),
        #         l_scale,
        #     ]
        # else:
        losses = [
            l_ce,
            l_reg,
            l_scale,
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
