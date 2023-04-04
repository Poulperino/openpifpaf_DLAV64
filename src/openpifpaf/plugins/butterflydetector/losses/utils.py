"""Losses."""

import logging
import torch
import math
import numpy as np
# from ..decoder.utils import index_field

LOG = logging.getLogger(__name__)



#@torch.jit.script
def ratio_iou_scripted(x1, y1, w1, h1, x2, y2, w2, h2):
    x1 = x1 - w1/2
    y1 = y1 - h1/2
    x2 = x2 - w2/2
    y2 = y2 - h2/2
    xi = torch.max(x1, x2)                                    # Intersection (yi similarly)
    yi = torch.max(y1, y2)                                    # Intersection (yi similarly)
    wi = torch.clamp(torch.min(x1+w1, x2+w2) - xi + 1, min=0, max=math.inf)
    hi = torch.clamp(torch.min(y1+h1, y2+h2) - yi + 1, min=0, max=math.inf)
    area_i = wi * hi                                      # Area Intersection
    area_u = (w1+1) * (h1+1) + (w2+1) * (h2+1) - wi * hi    # Area Union
    result = area_i / torch.clamp(area_u, min=1e-5, max=math.inf)
    try:
        assert((result>=0.).all() and (result<=1.0).all())
    except:
        import pdb; pdb.set_trace()
    return torch.clamp(result, min=1e-5, max=math.inf)

def ratio_siou_scripted(x1, y1, w1, h1, x2, y2, w2, h2):
    x1 = x1 - w1/2
    y1 = y1 - h1/2
    x2 = x2 - w2/2
    y2 = y2 - h2/2
    xi = torch.max(x1, x2)                                    # Intersection (yi similarly)
    yi = torch.max(y1, y2)
    wi = torch.min(x1+w1, x2+w2) - xi
    hi = torch.min(y1+h1, y2+h2) - yi
    mask = (wi>0) & (hi>0)
    area_i = - torch.abs(wi * hi)                                  # Area Intersection
    area_i[mask] = - area_i[mask]
    try:
        assert((w1>=0.).all() and (w2>=0).all() and (h1>=0).all() and (h2>=0).all())
    except:
        import pdb; pdb.set_trace()
    area_u = w1 * h1 + w2 * h2 - area_i    # Area Union
    result = area_i / area_u
    return result

def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    max_r = torch.min((torch.stack(max_r1, max_r2, max_r3, max_r4)), axis=0)
    m0 = torch.isfinite(max_r)
    x = x[:, m0]
    t = t[:, m0]
    max_r = max_r[m0]

    # m1 = (x - t).norm(p=1, dim=0) > max_r
    # x = x[:, m1]
    # t = t[:, m1]
    # max_r = max_r[m1]

    norm = (x - t).norm(dim=0)
    m2 = norm > max_r

    return torch.sum(norm[m2] - max_r[m2])


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[0, :] < 0.0] += 1
    q[xys[1, :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)

    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]

    return (
        torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) +
        torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) +
        torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) +
        torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])
    )


class SmoothL1Loss(object):
    def __init__(self, r_smooth, scale_required=True):
        self.r_smooth = r_smooth
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)

class MultiHeadLoss(torch.nn.Module):
    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses

class MultiHeadLossAutoTune(torch.nn.Module):
    def __init__(self, losses, lambdas):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        assert all(l >= 0.0 for l in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float32),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = np.array([lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                       if l is not None])
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                    if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if not(loss_values is None) else None

        return total_loss, flat_head_losses

class RepulsionLoss(torch.nn.Module):

    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()

    def calc_iou_pairwise(self, a, b):
        area = b[:, 2] * b[:, 3]

        iw = torch.min(torch.unsqueeze((a[:, 0] + a[:, 2]), dim=1), (b[:, 0] + b[:, 2])) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze((a[:, 1] + a[:, 3]), dim=1), (b[:, 1] + b[:, 3])) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.clamp(torch.unsqueeze(a[:, 2] * a[:, 3], dim=1) + area - iw * ih , min=1e-8)

        return (iw * ih) / ua

    def IoG(self, box_a, box_b):
        """Compute the IoG of two sets of boxes.
        E.g.:
            A ∩ B / A = A ∩ B / area(A)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_objects,4]
        Return:
            IoG: (tensor) Shape: [num_objects]
        """
        inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
        inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
        inter_xmax = torch.min((box_a[:, 0] + box_a[:, 2]), (box_b[:, 0] + box_b[:, 2]))
        inter_ymax = torch.min((box_a[:, 1] + box_a[:, 3]), (box_b[:, 1] + box_b[:, 3]))
        Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
        Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
        return (Iw * Ih) / (box_a[:, 2] * box_a[:, 3])

    # TODO
    def smooth_ln(self, x, smooth):
        return torch.where(
            torch.le(x, smooth),
            -torch.log(1 - x),
            ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
        )

    def forward(self, predict_boxes, ground_data, ids):

        RepGT_loss, RepBox_loss = torch.tensor(0).float().cuda(), torch.tensor(0).float().cuda()

        IoU = self.calc_iou_pairwise(ground_data, predict_boxes)
        IoU_argmax = torch.arange(len(ground_data)).cuda()
        positive_indices = torch.ge(IoU[IoU_argmax, IoU_argmax], 0.5)

        if positive_indices.sum() > 0:
            for index, elem in enumerate(ids):
                IoU[index, ids==elem] = -1
            _, IoU_argsec = torch.max(IoU, dim=1)
            IoG_to_minimize = self.IoG(ground_data[IoU_argsec, :], predict_boxes)
            RepGT_loss = self.smooth_ln(IoG_to_minimize, 0.5)
            RepGT_loss = RepGT_loss.mean()

            # add PepBox losses
            IoU_argmax_pos = IoU_argmax[positive_indices].float()
            IoU_argmax_pos = IoU_argmax_pos.unsqueeze(0).t()
            predict_boxes = torch.cat([predict_boxes, IoU_argmax_pos], dim=1)
            predict_boxes_np = predict_boxes.detach().cpu().numpy()
            num_gt = bbox_annotation.shape[0]
            predict_boxes_sampled = []
            for id in range(num_gt):
                index = np.where(predict_boxes_np[:, 4]==id)[0]
                if index.shape[0]:
                    idx = random.choice(range(index.shape[0]))
                    predict_boxes_sampled.append(predict_boxes[index[idx], :4])
            iou_repbox = self.calc_iou(predict_boxes_sampled, predict_boxes_sampled)
            iou_repbox = iou_repbox * mask
            RepBox_loss = self.smooth_ln(iou_repbox, 0.5)
            RepBox_loss = RepBox_loss.sum() / torch.clamp(torch.sum(torch.gt(iou_repbox, 0)).float(), min=1.0)
        return RepGT_loss, RepBox_loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.m = torch.nn.Sigmoid()
        self.previous=None

    def forward(self, preds, gt, bce_weight):
        pos_mask = gt.long().eq(1)
        neg_mask = gt.long().lt(1)
        res = self.m(preds)
        #res = torch.exp(res)
        #neg_weights = torch.pow(1 - gt[neg_mask], 4)
        pos_loss = self.alpha*torch.log(res[pos_mask]) * torch.pow(1 - res[pos_mask], self.gamma)
        neg_loss = (1-self.alpha)*torch.log(1 - res[neg_mask]) * torch.pow(res[neg_mask], self.gamma)#*neg_weights
        loss = torch.cat((pos_loss, neg_loss), 0)

        #self.previous = preds.clone()
        # if (gt==1).sum() != 0:
        #     return -self.alpha*loss/(gt==1).sum()
        # else:
        #     return -self.alpha*loss/1
        return - loss.sum()/max((gt==1).sum(),1.0)
        #import pdb; pdb.set_trace()
        #return torch.nn.functional.nll_loss(((1 - res) ** self.gamma) * log_res, gt.long(), bce_weight)
