#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  SAVE_IMAGES=$1; shift
  GPUS=($@)

  # best results on DAGM dataset
  train_DAGM $SAVE_IMAGES N_ALL $RESULTS_PATH 7 1000 70 0.05 1 1 True  1 10 True True  True  "${GPUS[@]}"

  # best results on KolektorSDD dataset
  train_KSDD $SAVE_IMAGES N_33   $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True True  True  "${GPUS[@]}"

  # best results on KolektorSDD2 dataset
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 10 True  2 3 True  True  True  ${GPUS[0]}

  # best results on Severstal Steel dataset
  train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_3000 $RESULTS_PATH 1 3000 3000 40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]}

}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"

