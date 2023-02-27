import argparse

import torch
import yaml

import openpifpaf
from ..utils.general import colorstr, check_img_size, check_dataset
from .dataset import LoadImagesAndLabels

class Yolov7DataLoader(openpifpaf.datasets.DataModule):
    # cli configurable
    data_dict = None
    rect = False
    cache_images = False
    single_cls = False
    image_weights = False
    quad = False
    imgsz = 640#, 640]
    # imgsz_test = 640 #[640, 640]
    upsample_stride = 1
    hyp = None
    device = torch.device('cpu')
    augmentation = True

    def __init__(self):
        super().__init__()
        check_dataset(self.data_dict)
        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'yolov7data', self.data_dict['names'])
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]
        # import pdb; pdb.set_trace()
        # self.imgsz = check_img_size(self.imgsz, self.head_metas[0].stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Yolov7 Datasets')

        group.add_argument('--yolov7-data', default=None, help='path to dataset config')
        group.add_argument('--yolov7-hyp', default=None, help='path to dataset config')
        group.add_argument('--yolov7-rect',
                           action='store_true', help='rectangular training')
        group.add_argument('--yolov7-cache-images',
                           action='store_true', help='cache images for faster training')
        group.add_argument('--yolov7-single-cls',
                           action='store_true', help='cache images for faster training')
        group.add_argument('--yolov7-image-weights',
                           action='store_true', help='use weighted image selection for training')
        group.add_argument('--yolov7-quad',
                           action='store_true', help='quad DataLoader')
        group.add_argument('--yolov7-img-size',
                           nargs='+', type=int, default=cls.imgsz, help='[train, test] image sizes')
        group.add_argument('--yolov7-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        if args.yolov7_data:
            with open(args.yolov7_data) as f:
                cls.data_dict = yaml.load(f, Loader=yaml.SafeLoader)

            with open(args.yolov7_hyp) as f:
                cls.hyp = yaml.load(f, Loader=yaml.SafeLoader)

        cls.rect = args.yolov7_rect
        cls.cache_images = args.yolov7_cache_images
        cls.single_cls = args.yolov7_single_cls
        cls.image_weights = args.yolov7_image_weights
        cls.quad = args.yolov7_quad
        cls.imgsz = args.yolov7_img_size
        cls.upsample_stride = args.yolov7_upsample
        cls.device = args.device

    # def _preprocess(self):
    #     enc = openpifpaf.encoder.CifDet(self.head_metas[0])
    #
    #     if not self.augmentation:
    #         return openpifpaf.transforms.Compose([
    #             openpifpaf.transforms.NormalizeAnnotations(),
    #             openpifpaf.transforms.RescaleAbsolute(self.square_edge),
    #             openpifpaf.transforms.CenterPad(self.square_edge),
    #             openpifpaf.transforms.EVAL_TRANSFORM,
    #             openpifpaf.transforms.Encoders([enc]),
    #         ])
    #
    #     if self.extended_scale:
    #         rescale_t = openpifpaf.transforms.RescaleRelative(
    #             scale_range=(0.5 * self.rescale_images,
    #                          2.0 * self.rescale_images),
    #             power_law=True, stretch_range=(0.75, 1.33))
    #     else:
    #         rescale_t = openpifpaf.transforms.RescaleRelative(
    #             scale_range=(0.7 * self.rescale_images,
    #                          1.5 * self.rescale_images),
    #             power_law=True, stretch_range=(0.75, 1.33))
    #
    #     return openpifpaf.transforms.Compose([
    #         openpifpaf.transforms.NormalizeAnnotations(),
    #         openpifpaf.transforms.RandomApply(
    #             openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
    #         rescale_t,
    #         openpifpaf.transforms.RandomApply(
    #             openpifpaf.transforms.Blur(), self.blur),
    #         openpifpaf.transforms.RandomChoice(
    #             [openpifpaf.transforms.RotateBy90(),
    #              openpifpaf.transforms.RotateUniform(10.0)],
    #             [self.orientation_invariant, 0.2],
    #         ),
    #         openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
    #         openpifpaf.transforms.CenterPad(self.square_edge),
    #         openpifpaf.transforms.MinSize(min_side=4.0),
    #         openpifpaf.transforms.UnclippedArea(threshold=0.75),
    #         openpifpaf.transforms.TRAIN_TRANSFORM,
    #         openpifpaf.transforms.Encoders([enc]),
    #     ])

    def _encoder(self):
        # if self.use_cifdet:
        #     enc = openpifpaf.encoder.CifDet(self.head_metas[0])
        # else:
        #     enc = ButterflyEncoder(self.head_metas[0])
        enc = openpifpaf.encoder.CifDet(self.head_metas[0])

        return openpifpaf.transforms.Encoders([enc])

    def train_loader(self):
        train_data = LoadImagesAndLabels(
            self.data_dict['train'], self.imgsz, self.batch_size,
            augment=True,  # augment images
            hyp=self.hyp,  # augmentation hyperparameters
            rect=self.rect,  # rectangular training
            cache_images=self.cache_images,
            single_cls=self.single_cls,
            stride=int(self.head_metas[0].stride),
            image_weights=self.image_weights,
            prefix=colorstr('train: '),
            encoder=self._encoder(),
            device=self.device
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = LoadImagesAndLabels(
            self.data_dict['val'], self.imgsz, self.batch_size,
            hyp=self.hyp,  # augmentation hyperparameters
            rect=True,  # rectangular training
            cache_images=self.cache_images,
            single_cls=self.single_cls,
            stride=int(self.head_metas[0].stride),
            image_weights=self.image_weights,
            pad=0.5,
            prefix=colorstr('val: '),
            encoder=self._encoder(),
            device=self.device
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    # @staticmethod
    # def _eval_preprocess():
    #     return openpifpaf.transforms.Compose([
    #         *CocoKp.common_eval_preprocess(),
    #         openpifpaf.transforms.ToAnnotations([
    #             openpifpaf.transforms.ToDetAnnotations(COCO_CATEGORIES),
    #             openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
    #         ]),
    #         openpifpaf.transforms.EVAL_TRANSFORM,
    #     ])

    def eval_loader(self):
        eval_data = LoadImagesAndLabels(
            self.data_dict['val'], self.imgsz, self.batch_size,
            hyp=None,  # augmentation hyperparameters
            rect=True,  # rectangular training
            cache_images=self.cache_images,
            single_cls=self.single_cls,
            stride=int(self.head_metas[0].stride),
            pad=0.5,
            prefix=colorstr('val: '),
            device=self.device
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        pass
