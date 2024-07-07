# Improved_Control_Points

## Quick Start
This is an improved document dewarping method based on [Control Points](https://github.com/gwxie/Document-Dewarping-with-Control-Points).
### clone

```
git clone https://github.com/NikLi66/Improved_Control_Points icp
cd icp
```


### distributed training
```
bash ddp_train.sh
```
Note: you could change the output path, data path and the number of gpus if necessary.

### test
```
bash test.sh
```

### evaluate
```
python eval.py
```

## Contributions
- Rewritting most part of the original codes to make it more readable.
- Adding CBAM modules.
- Adding Coord Conv modules.
- Adding some data augmentations widely used in dewarp task.

## Training on your own dataset
I strongly recommend you slightly change the codes in dataset/dataloader.py to make it works for your onw dataset. You only need to change the codes in __getitem__ and __init__ functions I suppose. The data format used in this project is following [Control Points](https://github.com/gwxie/Document-Dewarping-with-Control-Points). Please find more details in that repository if you like.

## Acknowledge
- Document Dewarping with Control Points [paper](https://arxiv.org/pdf/2203.10543) [github](https://github.com/gwxie/Document-Dewarping-with-Control-Points)
- CBAM: Convolutional Block Attention Module [paper](https://arxiv.org/abs/1807.06521)
- An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution [paper](https://arxiv.org/abs/1807.03247) [github](https://github.com/walsvid/CoordConv)
