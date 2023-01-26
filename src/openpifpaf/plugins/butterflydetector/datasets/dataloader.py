import argparse

import torch

import openpifpaf

from .constants import (
    UAVDT_CATEGORIES,
    VISDRONE_CATEGORIES,
    BBOX_KEYPOINTS,
    UAVDT_KEYPOINTS,
    HFLIP,
)

from .uavdt import UAVDT
from ..headmeta import Butterfly
from ..butterfly import Butterfly as ButterflyEncoder

class UAVDTDataLoader(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = "data/UAV-benchmark-M/GT/"
    val_annotations = "data/UAV-benchmark-M/GT/"
    eval_annotations = val_annotations
    train_image_dir = "data/UAV-benchmark-M/train/"
    val_image_dir = "data/UAV-benchmark-M/test/"
    eval_image_dir = "data/UAV-benchmark-M/test/"

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_annotation_filter = True

    use_cifdet = False

    def __init__(self):
        super().__init__()
        if self.use_cifdet:
            cifdet = openpifpaf.headmeta.CifDet('cifdet', 'uavdt', [UAVDT_CATEGORIES[0]])
        else:
            cifdet = Butterfly('butterfly', 'uavdt',
                              keypoints=UAVDT_KEYPOINTS,
                              categories=UAVDT_CATEGORIES)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module UAVDT')

        group.add_argument('--uavdt-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--uavdt-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--uavdt-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--uavdt-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--uavdt-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--uavdt-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--uavdt-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--uavdt-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--uavdt-no-augmentation',
                           dest='uavdt_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--uavdt-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--uavdt-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

        group.add_argument('--uavdt-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet head and encoder')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # uavdt specific
        cls.train_annotations = args.uavdt_train_annotations
        cls.val_annotations = args.uavdt_val_annotations
        cls.train_image_dir = args.uavdt_train_image_dir
        cls.val_image_dir = args.uavdt_val_image_dir

        cls.square_edge = args.uavdt_square_edge
        cls.extended_scale = args.uavdt_extended_scale
        cls.orientation_invariant = args.uavdt_orientation_invariant
        cls.blur = args.uavdt_blur
        cls.augmentation = args.uavdt_augmentation
        cls.rescale_images = args.uavdt_rescale_images
        cls.upsample_stride = args.uavdt_upsample

        cls.eval_annotation_filter = args.coco_eval_annotation_filter

        cls.use_cifdet = args.uavdt_cifdet

    def _preprocess(self):
        # enc = ButterflyEncoder(self.head_metas[0])
        if self.use_cifdet:
            enc = openpifpaf.encoder.CifDet(self.head_metas[0])
        else:
            enc = ButterflyEncoder(self.head_metas[0])
        if self.augmentation:
            preprocess_transformations = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.AnnotationJitter(),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, HFLIP), 0.5),
                openpifpaf.transforms.RescaleRelative(scale_range=(0.4 * self.rescale_images,
                                                        2.0 * self.rescale_images),
                                           power_law=True),
            ]
            if self.orientation_invariant:
                # preprocess_transformations += [
                #     openpifpaf.transforms.RotateBy90(),
                # ]
                preprocess_transformations += [openpifpaf.transforms.RandomChoice(
                    [openpifpaf.transforms.RotateBy90(),
                     openpifpaf.transforms.RotateUniform(10.0)],
                    [self.orientation_invariant, 0.2],),
                    ]
            preprocess_transformations += [
                openpifpaf.transforms.Crop(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
            ]
            preprocess_transformations += [
                openpifpaf.transforms.TRAIN_TRANSFORM,
            ]
            preprocess_transformations += [
                openpifpaf.transforms.Encoders([enc]),
            ]
        else:
            preprocess_transformations = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders([enc]),
            ]
        return openpifpaf.transforms.Compose(preprocess_transformations)

    def train_loader(self):
        train_data = UAVDT(
            root=self.train_image_dir,
            annFile=self.train_annotations,
            preprocess=self._preprocess(),
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet)
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle= not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):

        val_data =  UAVDT(
            root=self.val_image_dir,
            annFile=self.val_annotations,
            preprocess=self._preprocess(),
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet)
        )

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        if cls.batch_size == 1:
            preprocess = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.CenterPadTight(32),
            ]
        else:
            preprocess = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                openpifpaf.transforms.CenterPad(cls.eval_long_edge),
            ]
        return preprocess

    @staticmethod
    def _eval_preprocess():
        return openpifpaf.transforms.Compose([
            *UAVDTDataLoader.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToDetAnnotations(UAVDT_CATEGORIES),
                openpifpaf.transforms.ToCrowdAnnotations(UAVDT_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        data = UAVDT(
            root=self.eval_image_dir,
            annFile=None,
            preprocess=self._eval_preprocess(),
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet)
        )
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        pass
