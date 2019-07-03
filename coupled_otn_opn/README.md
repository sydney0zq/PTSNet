# Object Tracking Network coupled with Object Proposal Network

This directory contains code for generating coarse positions for each interested objects in DAVIS dataset. I have provided a sample tracking result in directory `result_davis`. And you could directly use it in DRSN.

<hr>

## How to run?

0. Prepare your dataset in `DAVIS/trainval` as described in the main page

1. Firstly, download the pretrained Object Tracking Network(OTN) and off-the-shelf Object Proposal Network(OPN) models from [OTN-GoogleDrive](https://drive.google.com/open?id=12bF1dRlEUZoQz3Qcr2WD3ojqNHzbCrjf) and [OPN-AmazonAWS](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl) respectively. Then put them into `models` and `tracking/maskrcnn/data` folders

2. If you have only one GPU card (GPUMemory > 6GB), you could simply run `python3 serial_infer.py val` to inference the whole validation set. Alternatively if you have multi-GPU cards, you could consider to exploit `local_run.sh` to speed up a lot. (Parallel)


## Our results

You can download our generated final tracking boxes from [coupled-otn-opn-result-davis-val](https://drive.google.com/open?id=1Z8Yn01IkjL1XuUH96Ble2IWwCVc3_9i-).

## Q & A

Q1: How could we train the model of OTN?

A1: You could modify the origin [py-MDNet](https://github.com/HyeonseobNam/py-MDNet) training code to achieve this(several lines). The provided model is obtained by changing the data source and keeping everything unchanged.


Q2: Why not merge OTN coupled OPN into DRSN?

A2: Actually we have done it. But for some reason, we prefer to release the disintegrated version to help others easy to understand and reproduce.


Q3: What is difference between the original MaskRCNN and your Object Proposal Network?

A3: Check the configurations in the `tracking/maskrcnn/configs` yaml files and we also add some functions to satify the needs of Object Tracking Network.

