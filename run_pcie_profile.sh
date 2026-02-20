#!/bin/bash
#nsys profile     -t cuda,nvtx     -o ./eval/pcie_transfer_profile/sift  --force-overwrite=true ./build/eval/pcie_transfer_profile sift
#
dataset=$1

sudo stdbuf -oL -eL nsys profile \
  --cuda-event-trace=false \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -o ./eval/pcie_transfer_profile/$dataset \
  ./build/eval/pcie_transfer_profile $dataset 2>&1 | tee eval/pcie_transfer_profile/${dataset}_energy.txt

  #--capture-range=nvtx \
  #--capture-range-end=stop \
  #--sample=none \
  #--cpuctxsw=none \
  #--backtrace=none \

