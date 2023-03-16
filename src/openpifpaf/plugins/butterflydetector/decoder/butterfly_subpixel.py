"""Decoder for pif-paf fields."""

import logging
import time
from typing import List
import argparse

from openpifpaf.decoder import Decoder
from . import generator
from .butterfly_hr import ButterflyHr
from .butterfly_seeds import ButterflySeeds
from ..utils import normalize_butterfly
from .. import headmeta
from ...yolov7.butterflyencoder import headmeta as headmeta_yolo
from .visualizer import Visualizer
LOG = logging.getLogger(__name__)


class Butterfly(Decoder):
    pif_fixed_scale = None
    scale_div = 10

    seed_threshold=0.2

    def __init__(self, head_metas: List[headmeta.Butterfly], *, visualizer=None):

        stride = [head_meta.stride for head_meta in head_metas]
        pif_index= list(range(len(head_metas)))
        head_names= [head_meta.name for head_meta in head_metas]
        pif_min_scale=0.0
        self.meta = head_metas[0]
        self.priority = -1.0  # prefer keypoints over detections
        self.priority += sum(m.n_fields for m in head_metas) / 1000.0

        self.strides = stride
        self.scale_wh = stride
        self.pif_min_scales = pif_min_scale
        # if 'nsbutterfly' in head_names:
        #     self.scale_wh = 1
        self.pif_indices = pif_index
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.pif_indices = [self.pif_indices]
        if not isinstance(self.pif_min_scales, (list, tuple)):
            self.pif_min_scales = [self.pif_min_scales for _ in self.strides]
        assert len(self.strides) == len(self.pif_indices)
        assert len(self.strides) == len(self.pif_min_scales)


        # self.seed_threshold = seed_threshold
        self.debug_visualizer = visualizer

        self.pif_nn = 16
        if 'obutterfly' in head_names:
            self.pif_nn_thres = 1

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        decoder_meta = []
        for meta in head_metas:
            if isinstance(meta, headmeta.Butterfly) or isinstance(meta, headmeta_yolo.Butterfly):
                decoder_meta.append(meta)

        return [
            Butterfly(decoder_meta, visualizer=Visualizer(decoder_meta[-1], file_prefix=cls.file_prefix))
        ]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        parser.add_argument('--debug-file-prefix', default=None,
                            help='save visualizer outputs')
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # check consistency
        cls.seed_threshold = args.seed_threshold
        cls.file_prefix = args.debug_file_prefix

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.strides, self.pif_indices):
                self.debug_visualizer.butterfly_raw(fields[pif_i], stride)

        #to numpy
        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)
        fields = apply(lambda x: x.numpy(), fields)
        # fields = [[field.cpu().numpy() for field in fields[0]]]

        # normalize
        normalized_pifs = [normalize_butterfly(*fields[pif_i], fixed_scale=self.pif_fixed_scale)[0]
                           for pif_i in self.pif_indices]

        # pif hr
        pifhr = ButterflyHr(self.pif_nn)
        pifhr.fill_sequence(normalized_pifs, self.strides, self.scale_wh, self.pif_min_scales)

        # seeds
        seeds = ButterflySeeds(pifhr, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer)
        seeds.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)

        # paf_scored

        gen = generator.Greedy(
            pifhr, seeds,
            meta=self.meta,
            seed_threshold=self.seed_threshold,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        # if self.force_complete:
        #     annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
