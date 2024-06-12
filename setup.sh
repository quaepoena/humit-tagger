#!/bin/sh
cat models/sentence_segmentation/x* > models/sentence_segmentation/pytorch_model.bin
cat models/classification/x* > models/classification/pytorch_model.bin
cat models/tokenization/x* > models/tokenization/pytorch_model.bin
rm models/*/x*
pip3 install -r requirements.txt
