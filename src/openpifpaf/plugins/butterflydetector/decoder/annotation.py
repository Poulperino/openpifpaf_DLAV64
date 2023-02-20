import numpy as np
import copy
import math
from openpifpaf import utils
# pylint: disable=import-error
from ..functional import scalar_value_clipped

class AnnotationDet(object):
    def __init__(self, j, xyv, n_joints, categories, dim_per_kps=3):
        self.categories = categories
        self.data = np.zeros((n_joints, dim_per_kps))
        self.joint_scales_w = None
        self.joint_scales_h = None
        self.data[j] = xyv
        self.bbox = np.array([xyv[0]-0.5*xyv[3], xyv[1]-0.5*xyv[4], xyv[3], xyv[4]])
        self.category_id = j + 1
        self.score = xyv[2]
        self.decoding_order = []

    def rescale(self, scale_factor):
        self.data[:, 0:2] *= scale_factor
        if self.joint_scales_w is not None:
            self.joint_scales_w *= scale_factor
        if self.joint_scales_h is not None:
            self.joint_scales_h *= scale_factor
        for _, __, c1, c2 in self.decoding_order:
            c1[:2] *= scale_factor
            c2[:2] *= scale_factor
        return self

    def fill_joint_scales(self, scales_w, scales_h, hr_scale=1.0):
        self.joint_scales_w = np.zeros((self.data.shape[0],))
        self.joint_scales_h = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale_field_w = scales_w[xyv_i]
            scale_field_h = scales_h[xyv_i]
            #i = max(0, min(scale_field_w.shape[1] - 1, int(round(xyv[0] * hr_scale))))
            #j = max(0, min(scale_field_w.shape[0] - 1, int(round(xyv[1] * hr_scale))))
            scale_w = scalar_value_clipped(scale_field_w, xyv[0] * hr_scale, xyv[1] * hr_scale)
            scale_h = scalar_value_clipped(scale_field_h, xyv[0] * hr_scale, xyv[1] * hr_scale)
            self.joint_scales_w[xyv_i] = scale_w / hr_scale
            self.joint_scales_h[xyv_i] = scale_h / hr_scale

    def fill_joint_scales_nothr(self, scales_w, scales_h, index, hr_scale=1.0):
        self.joint_scales_w = np.zeros((self.data.shape[0],))
        self.joint_scales_h = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            self.joint_scales_w[xyv_i] = scales_w[xyv_i][index] / hr_scale
            self.joint_scales_h[xyv_i] = scales_h[xyv_i][index] / hr_scale

    @property
    def category(self):
        return self.categories[self.category_id - 1]

    # def score(self):
    #     # v = self.data[:, 2]
    #     # return np.max(v)
    #     return self.score
    #     # return np.mean(np.square(v))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )

    def inverse_transform(self, meta):
        ann = copy.deepcopy(self)

        angle = -meta['rotation']['angle']
        if angle != 0.0:
            rw = meta['rotation']['width']
            rh = meta['rotation']['height']
            ann.bbox = utils.rotate_box(ann.bbox, rw - 1, rh - 1, angle)

        ann.bbox[:2] += meta['offset']
        ann.bbox[:2] /= meta['scale']
        ann.bbox[2:] /= meta['scale']

        if meta['hflip']:
            w = meta['width_height'][0]
            ann.bbox[0] = -(ann.bbox[0] + ann.bbox[2]) - 1.0 + w

        return ann
