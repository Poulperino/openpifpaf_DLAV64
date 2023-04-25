import dataclasses
import logging
from typing import ClassVar

import numpy as np
import scipy.ndimage
import torch

# from .annrescaler import AnnRescaler
# from ..utils import create_sink, mask_valid_area, create_sink_2d
from . import headmeta
from openpifpaf.utils import create_sink, mask_valid_area
from openpifpaf.encoder import AnnRescaler
from openpifpaf.visualizer import CifDet as CifDetVisualizer
# from .visualizer import Visualizer
from .utils import create_sink_2d

LOG = logging.getLogger(__name__)

@dataclasses.dataclass
class Butterfly:
    meta: headmeta.Butterfly
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifDetVisualizer = None

    side_length: ClassVar[int] = -2
    padding: ClassVar[int] = 10

    obutterfly: bool = False
    scale_wh: int = 1

    def __call__(self, image, anns, meta):
        return ButterflyGenerator(self)(image, anns, meta)

class ButterflyGenerator():
    def __init__(self, config: Butterfly):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride)
        self.visualizer = config.visualizer or CifDetVisualizer(config.meta)

        # self.sink = create_sink(config.side_length)
        # self.s_offset = (config.side_length - 1.0) / 2.0


        self.side_length = config.side_length
        self.v_threshold = config.v_threshold
        self.obutterfly = config.obutterfly
        self.stride = config.meta.stride
        self.scale_wh = config.scale_wh
        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.s_offset = None
        if self.side_length > -1:
            self.sink = create_sink(config.side_length)
            self.s_offset = ( config.side_length- 1.0) / 2.0
            if self.s_offset == 0:
                self.s_offset = (config.side_length- 1.0) / 2.0

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]
        n_fields = len(self.config.meta.keypoints)
        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask_detections = self.rescaler.bg_mask_detections(anns, width_height_original, n_fields,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)


        self.init_fields(n_fields, bg_mask_detections)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)
        self.visualizer.processed_image(image)

        self.visualizer.targets(torch.cat([
            fields[:,0:1],
            fields[:,1:3],
            torch.exp(fields[:,3:4]),
            torch.exp(fields[:,4:5]),
        ], axis=1), annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[-1] + 2 * self.config.padding
        field_h = bg_mask.shape[-2] + 2 * self.config.padding
        # self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_width = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        #self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_height = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        # self.intensities[-1] = 1.0
        # self.intensities[-1, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding] = bg_mask
        # self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
        #                                                     iterations=1 + 1,
        #                                                     border_value=1)
        # bg_mask_detections
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][bg_mask == 0] = np.nan

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[:, 2] > 0
        f = np.argmax(np.sum(np.reshape(visible, (-1, 5)), axis=1))
        if not np.any(visible):
            return

        width = (np.max(keypoints[visible, 0]*self.scale_wh) - np.min(keypoints[visible, 0]*self.scale_wh))
        height = (np.max(keypoints[visible, 1]*self.scale_wh) - np.min(keypoints[visible, 1]*self.scale_wh))
        area = (
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)/10
        LOG.debug('instance scale = %.3f', scale)
        xyv = keypoints[visible][-1]
        #f = len(keypoints)-1
        if xyv[2] <= self.v_threshold:
            return

        if self.side_length == -1:
            self.fill_coordinate_kps2(f, keypoints[visible], width, height, scale)
        elif self.side_length == -2:
            self.fill_coordinate_max4(f, keypoints[visible], width, height, scale)
        elif self.side_length == -3:
            self.fill_coordinate_kpsGradient(f, keypoints[visible], width, height, scale)
        else:
            self.fill_coordinate(f, xyv, width, height, scale)

    def fill_coordinate(self, f, xyv, width, height, scale):
        '''
        Use a normal 4x4 field
        '''
        #import pdb; pdb.set_trace()
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.side_length, miny + self.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return
        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update width and height
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)

    def fill_coordinate_max4(self, f, kps, width, height, scale):
        '''
        Use a 4x4 field with ignore region surrounding it.
        '''
        xy_offset = [-0.5, -0.5]
        #xy_offset = [0, 0]
        minx, miny = np.min(kps[kps[:,2]>0, 0]), np.min(kps[kps[:,2]>0, 1])
        maxx, maxy = np.max(kps[kps[:,2]>0, 0]), np.max(kps[kps[:,2]>0, 1])
        #w = np.round(w - xy_offset[0]).astype(np.int)
        #h = np.round(h - xy_offset[1]).astype(np.int)
        w = np.round(maxx - minx + 0.5).astype(np.int)

        h = np.round(maxy- miny + 0.5).astype(np.int)

        xyv = kps[-1]

        xy_offset = [(4 - 1.0) / 2.0, (4 - 1.0) / 2.0]

        ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.config.padding
        offset = xyv[:2] - (ij + xy_offset - self.config.padding)
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + 4, miny + 4

        if minx + 2 < 0 or maxx - 2 > self.intensities.shape[2] or \
           miny + 2 < 0 or maxy - 2 > self.intensities.shape[1]:
            return

        if w > 16 or h>16:
            sigma_ignore, sigma_eff = 0.5, 0.2
            w_ignore, h_ignore = np.round(sigma_ignore*w + 0.5).astype(np.int), np.round(sigma_ignore*h+ 0.5).astype(np.int)
            xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
            ij = np.round(xyv[:2] - xy_offset).astype(np.int) + self.config.padding
            if w>16:
                minx_ignore = int(ij[0])
            else:
                minx_ignore = minx
                w_ignore = 4
            if h>16:
                miny_ignore = int(ij[1])
            else:
                miny_ignore = miny
                h_ignore = 4
            maxx_ignore, maxy_ignore = minx_ignore + w_ignore, miny_ignore + h_ignore
            self.intensities[f, miny_ignore:maxy_ignore, minx_ignore:maxx_ignore] = np.nan
        w = 4
        h = 4

        sink = create_sink_2d(w, h)

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_width[f, miny:maxy, minx:maxx][mask] = np.log(width)
        self.fields_height[f, miny:maxy, minx:maxx][mask] = np.log(height)

    def fields(self, valid_area):
        intensities = self.intensities[:, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding]
        fields_reg = self.fields_reg[:, :, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding]
        fields_width = self.fields_width[:, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding]
        fields_height = self.fields_height[:, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding]
        #fields_scale = self.fields_scale[:, self.config.padding:-self.config.padding, self.config.padding:-self.config.padding]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_width, valid_area, fill_value=np.nan)
        mask_valid_area(fields_height, valid_area, fill_value=np.nan)

        # return (
        #     torch.from_numpy(intensities),
        #     torch.from_numpy(fields_reg),
        #     torch.from_numpy(fields_width),
        #     torch.from_numpy(fields_height),
        #     #torch.from_numpy(fields_scale),
        # )

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            np.expand_dims(fields_width, 1),
            np.expand_dims(fields_height, 1),
        ], axis=1))
