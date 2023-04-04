import argparse

import torch

import openpifpaf

from .constants import (
    UAVDT_CATEGORIES,
    VISDRONE_CATEGORIES,
    BBOX_KEYPOINTS,
    VISDRONE_BBOX_KEYPOINTS,
    UAVDT_BBOX_3CATEGORIES,
    UAVDT_KEYPOINTS,
    UAVDT_KEYPOINTS_3CATEGORIES,
    VISDRONE_KEYPOINTS,
    HFLIP,
    VISDRONE_HFLIP,
    UAVDT_HFLIP_3CATEGORIES
)

from .uavdt import UAVDT
from .visdrone import VisDrone
from ..headmeta import Butterfly, Butterfly_LaplaceWH
from ..butterfly import Butterfly as ButterflyEncoder
from .metric import AerialMetric
from ...butterfly2 import headmeta, encoder

class UAVDTDataLoader(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = "data/UAV-benchmark-M/GT/"
    val_annotations = "data/UAV-benchmark-M/GT/"
    eval_annotations = val_annotations
    train_image_dir = "data/UAV-benchmark-M/train/"
    val_image_dir = "data/UAV-benchmark-M/test/"
    # eval_image_dir = val_image_dir
    eval_image_dir = train_image_dir

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_annotation_filter = True

    use_cifdet = False
    use_3classes = False
    laplace_wh = False
    use_bf2 = False

    def __init__(self):
        super().__init__()
        if self.use_cifdet:
            cifdet = openpifpaf.headmeta.CifDet('cifdet', 'uavdt', [UAVDT_CATEGORIES[0]] if not self.use_3classes else UAVDT_CATEGORIES)
        elif self.laplace_wh:
            cifdet = Butterfly_LaplaceWH('butterfly_laplacewh', 'uavdt',
                              keypoints=UAVDT_KEYPOINTS if not self.use_3classes else UAVDT_KEYPOINTS_3CATEGORIES,
                              categories=UAVDT_CATEGORIES)
        elif self.use_bf2:
            cifdet = headmeta.Butterfly2('butterfly2', 'uavdt',
                            categories=[UAVDT_CATEGORIES[0]] if not self.use_3classes else UAVDT_CATEGORIES)
        else:
            cifdet = Butterfly('butterfly', 'uavdt',
                              keypoints=UAVDT_KEYPOINTS if not self.use_3classes else UAVDT_KEYPOINTS_3CATEGORIES,
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
        group.add_argument('--uavdt-3classes',
                           default=False, action='store_true',
                           help='Train to predict the 3 UAVDT classes')
        group.add_argument('--uavdt-laplacewh',
                           default=False, action='store_true',
                           help='Train WH using laplace')

        # Add BF2 support
        group.add_argument('--uavdt-bf2',
                           default=False, action='store_true',
                           help='Use Butterfly2 head and encoder')

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
        cls.use_3classes = args.uavdt_3classes
        cls.laplace_wh = args.uavdt_laplacewh
        cls.use_bf2 = args.uavdt_bf2

    def _preprocess(self):
        # enc = ButterflyEncoder(self.head_metas[0])
        if self.use_cifdet:
            enc = openpifpaf.encoder.CifDet(self.head_metas[0])
        elif self.use_bf2:
            enc = encoder.Butterfly2(self.head_metas[0])
        else:
            enc = ButterflyEncoder(self.head_metas[0])
        if self.augmentation:
            preprocess_transformations = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.AnnotationJitter(),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.HFlip(BBOX_KEYPOINTS if not self.use_3classes else UAVDT_BBOX_3CATEGORIES,
                                            HFLIP if not self.use_3classes else UAVDT_HFLIP_3CATEGORIES), 0.5),
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
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet),
            use_3classes=self.use_3classes
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
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet),
            use_3classes=self.use_3classes
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
            use_cifdet= isinstance(self.head_metas[0], openpifpaf.headmeta.CifDet),
            use_3classes=self.use_3classes
        )
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [AerialMetric(dataset='uavdt')]

class VisdroneDataLoader(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = "data/VisDrone2019/VisDrone2019-DET-train/annotations"
    val_annotations = "data/VisDrone2019/VisDrone2019-DET-val/annotations"
    eval_annotations = val_annotations
    train_image_dir = "data/VisDrone2019/VisDrone2019-DET-train/images"
    val_image_dir = "data/VisDrone2019/VisDrone2019-DET-val/images"
    eval_image_dir = val_image_dir

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_annotation_filter = True

    use_cifdet = False
    laplace_wh = False
    use_bf2 = False

    def __init__(self):
        super().__init__()
        if self.use_cifdet:
            cifdet = openpifpaf.headmeta.CifDet('cifdet', 'visdrone', VISDRONE_CATEGORIES)
        elif self.laplace_wh:
            cifdet = Butterfly_LaplaceWH('butterfly_laplacewh', 'visdrone',
                              keypoints=VISDRONE_KEYPOINTS,
                              categories=VISDRONE_CATEGORIES)
        elif self.use_bf2:
            cifdet = headmeta.Butterfly2('butterfly2', 'visdrone',
                            categories= VISDRONE_CATEGORIES)
        else:
            cifdet = Butterfly('butterfly', 'visdrone',
                              keypoints=VISDRONE_KEYPOINTS,
                              categories=VISDRONE_CATEGORIES)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Visdrone')

        group.add_argument('--visdrone-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--visdrone-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--visdrone-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--visdrone-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--visdrone-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--visdrone-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--visdrone-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--visdrone-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--visdrone-no-augmentation',
                           dest='visdrone_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--visdrone-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--visdrone-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

        group.add_argument('--visdrone-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet head and encoder')
        group.add_argument('--visdrone-laplacewh',
                           default=False, action='store_true',
                           help='Train WH using laplace')

        # Add BF2 support
        group.add_argument('--visdrone-bf2',
                           default=False, action='store_true',
                           help='Use Butterfly2 head and encoder')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # visdrone specific
        cls.train_annotations = args.visdrone_train_annotations
        cls.val_annotations = args.visdrone_val_annotations
        cls.train_image_dir = args.visdrone_train_image_dir
        cls.val_image_dir = args.visdrone_val_image_dir

        cls.square_edge = args.visdrone_square_edge
        cls.extended_scale = args.visdrone_extended_scale
        cls.orientation_invariant = args.visdrone_orientation_invariant
        cls.blur = args.visdrone_blur
        cls.augmentation = args.visdrone_augmentation
        cls.rescale_images = args.visdrone_rescale_images
        cls.upsample_stride = args.visdrone_upsample

        cls.eval_annotation_filter = args.coco_eval_annotation_filter

        cls.use_cifdet = args.visdrone_cifdet
        cls.laplace_wh = args.visdrone_laplacewh
        cls.use_bf2 = args.visdrone_bf2

    def _preprocess(self):
        # enc = ButterflyEncoder(self.head_metas[0])
        if self.use_cifdet:
            enc = openpifpaf.encoder.CifDet(self.head_metas[0])
        elif self.use_bf2:
            enc = encoder.Butterfly2(self.head_metas[0])
        else:
            enc = ButterflyEncoder(self.head_metas[0])
        if self.augmentation:
            preprocess_transformations = [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.AnnotationJitter(),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.HFlip(VISDRONE_BBOX_KEYPOINTS, VISDRONE_HFLIP), 0.5),
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
        train_data = VisDrone(
            root=self.train_image_dir,
            annFile=self.train_annotations,
            preprocess=self._preprocess(),
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle= not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):

        val_data =  VisDrone(
            root=self.val_image_dir,
            annFile=self.val_annotations,
            preprocess=self._preprocess(),
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
            *VisdroneDataLoader.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToDetAnnotations(VISDRONE_CATEGORIES),
                openpifpaf.transforms.ToCrowdAnnotations(VISDRONE_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        data = VisDrone(
            root=self.eval_image_dir,
            annFile=None,
            preprocess=self._eval_preprocess(),
        )
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [AerialMetric(dataset='visdrone')]
