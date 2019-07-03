#! /bin/sh
#
# Copyright (C) 2018 qiang.zhou <qiang.zhou@train119.hogpu.cc>
#
# Distributed under terms of the MIT license.
#


echo "Error: Please read the code logic before your training."
exit

PYTHON_EXE=python3

# Train on YouTube-VOS
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON_EXE train.py --model_save_path experiments/snapshots --max_iters 100000 --decayat 60000 --learning_rate 2e-5 --batch_size 64 --input_size 256,256

# Adapt training on DAVIS
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON_EXE train_adapt_davis.py --seg_model_path experiments/snapshots/drsn_100000.pth --steps 25000 --lr_s1 2e-7 --steps_s2 10000 --lr_s2 2e-8

# Move to snapshot directory
mv experiments/stage2/drsn_davis_10000.pth snapshots/drsn_yvos_10w_davis_3p5w.pth

# Evaluation
# One GPU
CUDA_VISIBLE_DEVICES=0 python3 segment_ft.py 1e-6 800
# Four GPUs
#CUDA_VISIBLE_DEVICES=0 python3 segment_wft.py 1e-6 800 aa &
#CUDA_VISIBLE_DEVICES=1 python3 segment_wft.py 1e-6 800 ab &
#CUDA_VISIBLE_DEVICES=2 python3 segment_wft.py 1e-6 800 ac &
#CUDA_VISIBLE_DEVICES=3 python3 segment_wft.py 1e-6 800 ad &
#wait


# J Mean, if you want to measure the F Mean, use the offical toolbox on https://github.com/davisvideochallenge/davis-matlab
python3 davis_eval.py result_davis_mask     
