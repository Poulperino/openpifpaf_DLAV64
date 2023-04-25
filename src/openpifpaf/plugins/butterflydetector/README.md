# Installation

```sh
1- Follow instructions to install the openpifpaf repo
2- cd src/openpifpaf/plugins/butterflydetector
3- Run python  functional_setup.py build_ext --inplace
```

# Training
```sh
python -m openpifpaf.train --lr=2e-3 --b-scale=10.0 --momentum=0.95 \
  --epochs=150 --lr-decay 120 140 \
  --batch-size=5 --weight-decay=1e-5 --basenet=hrnetw32det \
  --dataset visdrone --visdrone-square-edge 512 --visdrone-orientation-invariant=0.1


python -m openpifpaf.train --lr=2e-3 --b-scale=10.0 --momentum=0.95 \
  --epochs=70 --lr-decay 40 60 \
  --batch-size=5 --weight-decay=1e-5 --basenet=hrnetw32det \
  --dataset uavdt --uavdt-square-edge 512 --uavdt-orientation-invariant=0.1
```

# Testing
```sh
1- the same command used for openpifpaf should work
2- It will save the predictions in a folder set by --output
```
