import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from .. import transforms, utils

import pandas as pd
import glob

LOG = logging.getLogger(__name__)

class MOT(torch.utils.data.Dataset):
    """`UAVDT <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    train_image_dir = "data/UAV-benchmark-M/train/"
    val_image_dir = "data/UAV-benchmark-M/test/"
    train_annotations = "data/UAV-benchmark-M/GT/"
    val_annotations = "data/UAV-benchmark-M/GT/"
    test_path = {'val': "data/UAV-benchmark-M/test/"}
    def __init__(self, root, annFile, *, target_transforms=None, n_images=None, preprocess=None):
        self.root = root
        folders = os.listdir(self.root)

        self.imgs = []
        self.targets = []

        for seq_id in folders:
            img_ids = sorted(os.listdir(os.path.join(self.root, seq_id, 'img1')))
            image_paths = map(lambda img_name: os.path.join(self.root, seq_id, 'img1', img_name), img_ids)
            df = pd.read_csv(os.path.join(self.root, seq_id, 'gt', 'gt.txt'), sep=',', header=None)
            grouped = dict(tuple(df.groupby(df.columns[0])))#df.reset_index().groupby(0)['index'].apply(list).sort_values().tolist()
            self.imgs.extend(list(image_paths))
            self.targets.extend(list(grouped.values()))

        self.imgs = np.asarray(self.imgs)
        self.targets = np.asarray(self.targets)
        print('Images: {}'.format(len(self.imgs)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms
        self.categ2idx = {1:1, 2:1, 3:2, 4:3, 5:4, 6:5, 7:1, 8:8, 9:9, 10:10, 11:11, 12:12}
        self.uniqueClasses = [1,2,3,4,5]
        self.log = logging.getLogger(self.__class__.__name__)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = self.imgs[index]
        with open(os.path.join(img_path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        initial_size = image.size
        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_name': os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]),
        }
        anns = []
        for target in self.targets[index].to_numpy():
            w = target[4]
            h = target[5]
            x = target[2]
            y = target[3]
            categ = self.categ2idx[target[7]]
            if categ in [9,10]:
                continue
            if categ in [8,12]:
                for ignore_class in self.uniqueClasses:
                    anns.append({
                        'image_id': index,
                        'category_id': ignore_class,
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "iscrowd": 1,
                        "segmentation":[],
                        'keypoints': [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        'num_keypoints': 5,
                    })
            else:
                anns.append({
                    'image_id': index,
                    'category_id': categ,
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 1-int(target[6]),
                    "segmentation":[],
                    'keypoints': [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    'num_keypoints': 5,
                })

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        self.log.debug(meta)
        # transform targets
        if self.target_transforms is not None:
            width_height = image.shape[2:0:-1]
            anns = [t(anns, width_height) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.imgs)

    def write_evaluations(self, eval_class, path, total_time):
        for folder in eval_class.dict_folder.keys():
            utils.mkdir_if_missing(path)
            with open(os.path.join(path,folder+".txt"), "w") as file:
                file.write("\n".join(eval_class.dict_folder[folder]))
        n_images = len(eval_class.image_ids)

        print('n images = {}'.format(n_images))
        print('decoder time = {:.1f}s ({:.0f}ms / image)'
              ''.format(eval_class.decoder_time, 1000 * eval_class.decoder_time / n_images))
        print('total time = {:.1f}s ({:.0f}ms / image)'
              ''.format(total_time, 1000 * total_time / n_images))

class MOT17(MOT):
    train_image_dir = "data/MOT17Det/train/"
    val_image_dir = "data/MOT17Det/train/"
    train_annotations = None
    val_annotations = None
    test_path = {'val': None}

class MOT20(MOT):
    train_image_dir = "data/MOT20Det/train/"
    val_image_dir = "data/MOT20Det/train/"
    train_annotations = None
    val_annotations = None
    test_path = {'val': None}
