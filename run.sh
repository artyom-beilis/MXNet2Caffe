#!/bin/bash

SHAPE=128,3,32,32
SRC_DIR=~/Packages/mxnet/mxnet/example/image-classification/
# SRC_NET=trained-lenet.json
# SRC_MOD=trained-lenet.model
SRC_NET=trained-cifar10.json
SRC_MOD=trained-cifar10.model

TGT_DIR=/home/artik/Projects/makeahmap/erode
TGT_NET=test.prototxt
TGT_MOD=test.deploy


python json2prototxt.py --shape $SHAPE --mx-json $SRC_DIR/$SRC_NET                              --cf-prototxt $TGT_DIR/$TGT_NET                                 || exit 1
python mxnet2caffe.py   --shape $SHAPE --mx-json $SRC_DIR/$SRC_NET --mx-model $SRC_DIR/$SRC_MOD --cf-prototxt $TGT_DIR/$TGT_NET --cf-model $TGT_DIR/$TGT_MOD    || exit 1
