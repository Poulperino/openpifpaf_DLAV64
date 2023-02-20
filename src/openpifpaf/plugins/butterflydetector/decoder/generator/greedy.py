from collections import defaultdict
import logging
import time
import torchvision
import torch
import numpy as np

from ..annotation import AnnotationDet as Annotation
from ...utils import scalar_square_add_2dsingle

# pylint: disable=import-error
from ...functional import scalar_nonzero
from ..nms import non_max_suppression_fast

LOG = logging.getLogger(__name__)


class Greedy(object):

    # suppression = 0.1
    # iou_threshold = 0.5
    suppression = 0.0
    iou_threshold = 0.8
    instance_threshold = 0.05
    nms_by_category = True

    def __init__(self, pifhr, seeds, meta, *,
                 seed_threshold,
                 debug_visualizer=None):
        self.pifhr = pifhr
        self.seeds = seeds
        self.meta = meta
        self.seed_threshold = seed_threshold

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        if self.debug_visualizer:
            self.debug_visualizer.butterfly_hr(self.pifhr.targets)
            self.debug_visualizer.butterflyhr_wh(self.pifhr.widths, self.pifhr.heights, self.pifhr.targets)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        occupied = np.zeros(self.pifhr.scales.shape, dtype=np.uint8)
        annotations = []

        def mark_occupied(ann):
            try:
                for joint_i, xyv in enumerate(ann.data):
                    if xyv[2] == 0.0:
                        continue

                    width = max(4, ann.joint_scales_w[joint_i])
                    height = max(4, ann.joint_scales_h[joint_i])
                    scalar_square_add_2dsingle(occupied[joint_i],
                                             xyv[0],
                                             xyv[1],
                                             width / 2.0, height/2.0, 1)
                                             #np.clip(xyv[3]/5,a_min=2, a_max=None) / 2.0, np.clip(xyv[4]/5,a_min=2, a_max=None)/2, 1)
            except:
                import pdb; pdb.set_trace()

        for ann in initial_annotations:
            if ann.joint_scales is None:
                ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            annotations.append(ann)
            mark_occupied(ann)

        boxes = []
        scores = []
        categories = []
        for v, f, x, y, w, h in self.seeds.get():
            if scalar_nonzero(occupied[f], x, y):
                continue
            boxes.append([x-0.5*w, y-0.5*h, w, h])
            scores.append(v)
            categories.append(f+1)
            ann = Annotation(f, (x, y, v, w, h), self.pifhr.scales_w.shape[0], categories=self.meta.categories, dim_per_kps=5)
            ann.fill_joint_scales(self.pifhr.scales_w, self.pifhr.scales_h)
            annotations.append(ann)
            mark_occupied(ann)

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        categories = np.asarray(categories)
        # # if new_nms:
        # if self.nms_by_category:
        #     keep_index = torchvision.ops.batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(categories), self.iou_threshold)
        # else:
        #     keep_index = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.iou_threshold)
        # # else:
        # #     bboxes_res = np.array([])
        # #     scores_res = np.array([])
        # #     if self.args.snms:
        # #         for cls in set(categories):
        # #             pick, scores_temp = py_cpu_softnms(bboxes[categories==cls], scores[categories==cls], Nt=0.5, sigma=0.5, thresh=self.args.snms_threshold, method=1)
        # #             if len(bboxes_res) == 0:
        # #                 bboxes_res = bboxes[categories==cls][pick]
        # #                 scores_res = scores_temp
        # #             else:
        # #                 bboxes_res = np.concatenate((bboxes_res, bboxes[categories==cls][pick]))
        # #                 scores_res = np.concatenate((scores_res, scores_temp))
        # #     elif self.args.nms:
        # #         _, pick = non_max_suppression_fast(np.concatenate((bboxes, scores[:, np.newaxis]), axis=1), categories, overlapThresh=self.args.nms_threshold)
        # #         bboxes_res = bboxes[pick]
        # #         scores_res = scores[pick]
        #
        if len(annotations)>0:
            _, pick = non_max_suppression_fast(np.concatenate((boxes, scores[:, np.newaxis]), axis=1), categories, overlapThresh=0.8)
            annotations = np.asarray(annotations)[pick].tolist()
            # scores = scores[pick]

        # keep_index = keep_index.numpy()
        # pre_nms_scores = np.copy(scores)
        # scores *= self.suppression
        # scores[keep_index] = pre_nms_scores[keep_index]
        # filter_mask = (scores > self.instance_threshold)
        # categories = categories[filter_mask]
        # scores = scores[filter_mask]
        # boxes = boxes[filter_mask]
        # annotations = np.asarray(annotations)[filter_mask]
        #
        #
        #
        # for ann, score in zip(annotations, scores):
        #     ann.score = score

        annotations = sorted(annotations, key=lambda x: x.score, reverse=True)
        LOG.debug('cpp annotations = %d (%.1fms)',
                  len(scores),
                  (time.perf_counter() - start) * 1000.0)


        if self.debug_visualizer:
            LOG.debug('occupied field 0')
            self.debug_visualizer.occupied(occupied[0])

        LOG.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
