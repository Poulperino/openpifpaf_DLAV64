# Installation

```sh
1- Follow instructions to install the openpifpaf repo
2- Follow instruction from Yolov7 to set up the dataset
3- Modify cfg files inside ./dataloader/cfg/* to refer to the correct location of the dataset
4- If needed, modify the anchors in the cfg files in ./model/cfg/*
```

# Training
```sh
# For training using ButterflyDetector
python -m openpifpaf.train --lr=1e-4 --b-scale=10.0 \
--epochs=60 --lr-decay 40 50 \
--batch-size=32 --weight-decay=1e-5 --basenet=yolov7 \
--dataset yolov7data --yolov7-data src/openpifpaf/plugins/yolov7/dataloader/cfg/coco.yaml \
--yolov7-hyp src/openpifpaf/plugins/yolov7/dataloader/cfg/hyp.scratch.p5.yaml  --yolov7-fpn 3 \
--yolov7-fpn-relative-strides 4 2 1 --yolov7-fpn-largest-interval 256


# For training using Cifdet
python -m openpifpaf.train --lr=1e-4 --b-scale=10.0 \
--epochs=60 --lr-decay 40 50 \
--batch-size=16 --weight-decay=1e-5 --basenet=yolov7 \
--dataset yolov7data --yolov7-data src/openpifpaf/plugins/yolov7/dataloader/cfg/coco.yaml \
--yolov7-hyp src/openpifpaf/plugins/yolov7/dataloader/cfg/hyp.scratch.p5.yaml  --yolov7-fpn 3 \
--yolov7-fpn-relative-strides 4 2 1 --yolov7-cifdet --yolov7-fpn-largest-interval 256

# For training without FPN
python -m openpifpaf.train --lr=1e-4 --b-scale=10.0 \
--epochs=150 --lr-decay 120 140 \
--batch-size=5 --weight-decay=1e-5 --basenet=hrnetw32det \
--dataset yolov7data --yolov7-data src/openpifpaf/plugins/yolov7/dataloader/cfg/visdrone.yaml \
--yolov7-hyp src/openpifpaf/plugins/yolov7/dataloader/cfg/hyp.scratch.p5.yaml

```

# Testing
```sh
1- The same command used for openpifpaf should work and with the different yolov7 and dataset arguments set correctly
2- The metric inside ./dataloader/dataloader.py needs to be modified for the coco dataset.
```
