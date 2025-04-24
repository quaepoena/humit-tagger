#!/bin/bash

ls models/*/x* &>/dev/null

if [[ "$?" -eq 0 ]]; then
    cat models/sentence_segmentation/x* > models/sentence_segmentation/pytorch_model.bin
    cat models/classification/x* > models/classification/pytorch_model.bin
    cat models/tokenization/x* > models/tokenization/pytorch_model.bin
    #rm models/*/x*
fi

pip3 install -r requirements.txt
