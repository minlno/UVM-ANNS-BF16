#/bin/bash

dataset=$1
topk=$2
batch_size=$3
nsys stats --filter-nvtx search_tk${topk}_bs${batch_size} ./${dataset}.nsys-rep
