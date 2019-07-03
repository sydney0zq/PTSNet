## Dynamic Reference Segmentation Network

This directory contains code for segmentation(pixel-level tracking). After you run over the coupled-otn-opn, you could put the `result_davis` into dataset folder. Note it contains training and testing phase. For training, we use YouTube-VOS Train set, and then use DAVIS 2017 Train set for adaption. For testing, we do the experiment on DAVIS 2017 validation set.


## How to run

1. Prepare the coupled OTN-OPN generated boxes into dataset folder.

2. Run `do_train_eval.sh`. Before you execute it, you need to modify this script for your machine.

3. You may be have interested in `visulization`, that can overlay masks on the images.



## Q and A

Q1: How long the training phase takes?
A1: On YouTube-VOS (2 days) and then on DAVIS (10 hours). But we didn't experiment the search of hyper-paramter.Maybe you could yield a better result at your settings.
