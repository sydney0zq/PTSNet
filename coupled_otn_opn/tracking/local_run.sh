#! /bin/sh
#
# local_run.sh
# Copyright (C) 2018 qiang.zhou <qiang.zhou@train119.hogpu.cc>
#
# Distributed under terms of the MIT license.
#

CUDA_VISIBLE_DEVICES=$1 python3 infer.py aa &
CUDA_VISIBLE_DEVICES=$2 python3 infer.py ab &
CUDA_VISIBLE_DEVICES=$3 python3 infer.py ac &
CUDA_VISIBLE_DEVICES=$4 python3 infer.py ad &
wait


