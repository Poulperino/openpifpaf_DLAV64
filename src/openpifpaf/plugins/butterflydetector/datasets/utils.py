import os
import copy

def is_non_zero_file(fpath):
    return True if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else False

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class PIF_Category(object):
    def __init__(self, num_classes, catID_label):
        self.num_classes = num_classes
        self.catID_label = catID_label

    def __call__(self, anns):
        for ann in anns:
            if ann['iscrowd']!=1:
                temp_id = self.catID_label[ann['category_id']]
                x, y, w, h = ann['bbox']
                temp = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(temp_id)\
                 + [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]\
                 + [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(self.num_classes-(temp_id+1))
                ann['keypoints'] = copy.deepcopy(temp)
                ann['num_keypoints'] = 5
        return anns
