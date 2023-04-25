import os
import numpy as np

from collections import defaultdict
from . import utils

class AerialMetric:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dict_folder = defaultdict(list)

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        """For every image, accumulate that image's predictions into this metric.

        :param predictions: List of predictions for one image.
        :param image_meta: Meta dictionary for this image as returned by the data loader.
        :param ground_truth: Ground truth information as produced by the eval
            loader. Optional because some metrics (i.e. pycocotools) read
            ground truth separately.
        """
        bboxes = defaultdict(list)
        fileName = image_meta['file_name']
        split_fileName = fileName.split("/")
        if self.dataset == "visdrone":
            folder = os.path.splitext(split_fileName[-1])[0]
        elif self.dataset == "uavdt":
            folder = split_fileName[-2]
            image_numb = int(split_fileName[-1][3:9])
        else:
            raise

        for ann in predictions:
            x, y, w, h = ann.bbox
            if w==0 or h==0:
                continue
            s = ann.score
            bboxes[ann.category_id].append([x, y, w, h, s])

        for categ in bboxes.keys():
            for bbox in np.array(bboxes[categ]):
                x, y, w, h, s = bbox
                if self.dataset == "visdrone":
                    self.dict_folder[folder].append(",".join(list(map(str,[x, y, w, h, s, categ, -1, -1]))))
                elif self.dataset == "uavdt":
                    self.dict_folder[folder].append(",".join(list(map(str,[image_numb, -1, x, y, w, h, s, 1, categ]))))

        if len(self.dict_folder[folder])==0:
            self.dict_folder[folder].append(",".join(list(map(str,[0, 0, 0, 0, 0, 0, -1, -1]))))

    def stats(self):
        """Return a dictionary of summary statistics.

        The dictionary should be of the following form and can contain
        an arbitrary number of entries with corresponding labels:

        .. code-block::

            {
                'stats': [0.1234, 0.5134],
                'text_labels': ['AP', 'AP0.50'],
            }
        """
        return {
            'pass': ['pass']
        }
        # raise NotImplementedError

    def write_predictions(self, path, *, additional_data=None):
        """Write predictions to a file.

        This is used to produce a metric-compatible output of predictions.
        It is used for test challenge submissions where a remote server
        holds the private test set.

        :param filename: Output filename of prediction file.
        :param additional_data: Additional information that might be worth saving
            along with the predictions.
        """
        for folder in self.dict_folder.keys():
            utils.mkdir_if_missing(path)
            with open(os.path.join(path, folder+".txt"), "w") as file:
                file.write("\n".join(self.dict_folder[folder]))
