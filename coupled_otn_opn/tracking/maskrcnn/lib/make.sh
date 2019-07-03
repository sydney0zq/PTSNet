#!/usr/bin/env bash
set -x

CUDA_PATH=/usr/local/cuda-9.2/
PYTHON_EXE=python3
NVCC=/usr/local/cuda-9.2/bin/nvcc

$PYTHON_EXE setup.py build_ext --inplace
/bin/rm -rf build

# Choose cuda arch as you need
CUDA_ARCH="-gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_61,code=sm_61 \
           -gencode arch=compute_70,code=sm_70 "
# Choose cuda arch as you need
#CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
#           -gencode arch=compute_35,code=sm_35 \
#           -gencode arch=compute_50,code=sm_50 \
#           -gencode arch=compute_52,code=sm_52 \
#           -gencode arch=compute_60,code=sm_60 \
#           -gencode arch=compute_61,code=sm_61 "
#          -gencode arch=compute_70,code=sm_70 "

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
$NVCC -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
$PYTHON_EXE build.py
#exit

# compile roi_pooling
cd ../../
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
$NVCC -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
$PYTHON_EXE build.py

## compile roi_align
#cd ../../
#cd model/roi_align/src
#echo "Compiling roi align kernels by nvcc..."
#nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
#	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
#cd ../
#python build.py

# compile roi_crop
cd ../../
cd model/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
$NVCC -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
$PYTHON_EXE build.py

# compile roi_align (based on Caffe2's implementation)
cd ../../
cd modeling/roi_xfrom/roi_align/src
echo "Compiling roi align kernels by nvcc..."
$NVCC -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
$PYTHON_EXE build.py
