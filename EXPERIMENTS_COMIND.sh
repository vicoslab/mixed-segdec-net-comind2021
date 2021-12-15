#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh

run_COMIND_EXPERIMENTS()
{
    RESULTS_PATH=$1; shift
    SAVE_IMAGES=$1; shift
    GPUS=($@)


    train_DAGM $SAVE_IMAGES N_0   $RESULTS_PATH 7 0    70 0.05 1 1 True  1 10 False True  True  "${GPUS[@]}" # Table 3, Figure 5, Table 4-Row 13
    train_DAGM $SAVE_IMAGES N_5   $RESULTS_PATH 7 5    70 0.05 1 1 True  1 10 True  True  True  "${GPUS[@]}" # Table 3, Figure 5
    train_DAGM $SAVE_IMAGES N_15  $RESULTS_PATH 7 15   70 0.05 1 1 True  1 10 True  True  True  "${GPUS[@]}" # Table 3, Figure 5, Table 4-Row 12
    train_DAGM $SAVE_IMAGES N_45  $RESULTS_PATH 7 45   70 0.05 1 1 True  1 10 True  True  True  "${GPUS[@]}" # Table 3, Figure 5
    train_DAGM $SAVE_IMAGES N_ALL $RESULTS_PATH 7 1000 70 0.05 1 1 True  1 10 True  True  True  "${GPUS[@]}" # Table 3, Figure 5


    train_KSDD $SAVE_IMAGES N_0    $RESULTS_PATH 7 33 0  50 0.01 1 1 True  2 1 False True  True  "${GPUS[@]}" # Figure 7, Table 4-Row 13
    train_KSDD $SAVE_IMAGES N_5    $RESULTS_PATH 7 33 5  50 1 0.01 1 True  2 1 True  True  True  "${GPUS[@]}" # Figure 7, Table 4-Row 12
    train_KSDD $SAVE_IMAGES N_10   $RESULTS_PATH 7 33 10 50 1 0.01 1 True  2 1 True  True  True  "${GPUS[@]}" # Figure 7
    train_KSDD $SAVE_IMAGES N_15   $RESULTS_PATH 7 33 15 50 1 0.01 1 True  2 1 True  True  True  "${GPUS[@]}" # Figure 7
    train_KSDD $SAVE_IMAGES N_20   $RESULTS_PATH 7 33 20 50 1 0.01 1 True  2 1 True  True  True  "${GPUS[@]}" # Figure 7
    train_KSDD $SAVE_IMAGES N_ALL  $RESULTS_PATH 7 33 33 50 1 0.01 1 True  2 1 True  True  True  "${GPUS[@]}" # Figure 7


    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_0   $RESULTS_PATH 15 -1 0   50 0.01 1 1 True  2 3 False True  True  ${GPUS[0]} # Figure 9, Table 4-Row 13
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_16  $RESULTS_PATH 15 -1 16  50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9, Table 4-Row 12
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_53  $RESULTS_PATH 15 -1 53  50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_126 $RESULTS_PATH 15 -1 126 50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9


    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_0     $RESULTS_PATH 1 300  0    90 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_10    $RESULTS_PATH 1 300  10   90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_50    $RESULTS_PATH 1 300  50   90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_150   $RESULTS_PATH 1 300  150  90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_300_N_300   $RESULTS_PATH 1 300  300  90 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12

    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_0     $RESULTS_PATH 1 750  0    80 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_10    $RESULTS_PATH 1 750  10   80 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_50    $RESULTS_PATH 1 750  50   80 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_150   $RESULTS_PATH 1 750  150  80 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_300   $RESULTS_PATH 1 750  300  80 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_750_N_750   $RESULTS_PATH 1 750  750  80 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12

    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_0    $RESULTS_PATH 1 1500 0    60 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_10   $RESULTS_PATH 1 1500 10   60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_50   $RESULTS_PATH 1 1500 50   60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_150  $RESULTS_PATH 1 1500 150  60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_300  $RESULTS_PATH 1 1500 300  60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_750  $RESULTS_PATH 1 1500 750  60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_1500_N_1500 $RESULTS_PATH 1 1500 1500 60 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12

    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_0    $RESULTS_PATH 1 3000 0    40 0.1 0.1 10 True  2 1 False True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_10   $RESULTS_PATH 1 3000 10   40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_50   $RESULTS_PATH 1 3000 50   40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_150  $RESULTS_PATH 1 3000 150  40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_300  $RESULTS_PATH 1 3000 300  40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_750  $RESULTS_PATH 1 3000 750  40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_1500 $RESULTS_PATH 1 3000 1500 40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12
    train_single $SAVE_IMAGES STEEL $STEEL_PATH ALL_3000_N_3000 $RESULTS_PATH 1 3000 3000 40 0.1 0.1 10 True  2 1 True  True  True  ${GPUS[0]} # Figure 12

    train_DAGM $SAVE_IMAGES FS_ROW1   $RESULTS_PATH 7 1000  70 0.05 1 1 False 1 10 False False True  "${GPUS[@]}" # Table 4, Row 1
    train_DAGM $SAVE_IMAGES FS_ROW2   $RESULTS_PATH 7 1000  70 0.05 1 1 False 1 10 True  False True  "${GPUS[@]}" # Table 4, Row 2
    train_DAGM $SAVE_IMAGES FS_ROW3   $RESULTS_PATH 7 1000  70 0.05 1 1 False 1 10 True  True  True  "${GPUS[@]}" # Table 4, Row 3
    train_DAGM $SAVE_IMAGES FS_ROW4   $RESULTS_PATH 7 1000  70 0.05 1 1 True  1 10 True  False True  "${GPUS[@]}" # Table 4, Row 4
    train_DAGM $SAVE_IMAGES FS_ROW5   $RESULTS_PATH 7 1000  70 0.05 1 1 True  1 10 False True  True  "${GPUS[@]}" # Table 4, Row 5
    # FS_ROW6 is N_ALL from above

    train_DAGM $SAVE_IMAGES MS_ROW1   $RESULTS_PATH 7 15  70 0.05 1 1 False 1 10 False False True  "${GPUS[@]}" # Table 4, Row 7
    train_DAGM $SAVE_IMAGES MS_ROW2   $RESULTS_PATH 7 15  70 0.05 1 1 False 1 10 True  False True  "${GPUS[@]}" # Table 4, Row 8
    train_DAGM $SAVE_IMAGES MS_ROW3   $RESULTS_PATH 7 15  70 0.05 1 1 False 1 10 True  True  True  "${GPUS[@]}" # Table 4, Row 9
    train_DAGM $SAVE_IMAGES MS_ROW4   $RESULTS_PATH 7 15  70 0.05 1 1 True  1 10 True  False True  "${GPUS[@]}" # Table 4, Row 10
    train_DAGM $SAVE_IMAGES MS_ROW5   $RESULTS_PATH 7 15  70 0.05 1 1 True  1 10 False True  True  "${GPUS[@]}" # Table 4, Row 11
    # MS_ROW6 is N_15 from above

    # WS_ROW1 is N_0 from above
    train_DAGM $SAVE_IMAGES WS_ROW2   $RESULTS_PATH 7 15  70 0.05 1 1 True  1 10 True True  True  "${GPUS[@]}" # Table 4, Row 14



    train_KSDD $SAVE_IMAGES FS_ROW1   $RESULTS_PATH 7 33 33  50 1 0.01 1 False 2 1 False False True  "${GPUS[@]}" # Table 4, Row 1
    train_KSDD $SAVE_IMAGES FS_ROW2   $RESULTS_PATH 7 33 33  50 1 0.01 1 False 2 1 True  False True  "${GPUS[@]}" # Table 4, Row 2
    train_KSDD $SAVE_IMAGES FS_ROW3   $RESULTS_PATH 7 33 33  50 1 0.01 1 False 2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 3
    train_KSDD $SAVE_IMAGES FS_ROW4   $RESULTS_PATH 7 33 33  50 1 0.01 1 True  2 1 True  False True  "${GPUS[@]}" # Table 4, Row 4
    train_KSDD $SAVE_IMAGES FS_ROW5   $RESULTS_PATH 7 33 33  50 1 0.01 1 True  2 1 False True  True  "${GPUS[@]}" # Table 4, Row 5
    # FS_ROW6 is N_ALL from above

    train_KSDD $SAVE_IMAGES MS_ROW1   $RESULTS_PATH 7 33 5  50 1 0.01 1 False 2 1 False False True  "${GPUS[@]}" # Table 4, Row 7
    train_KSDD $SAVE_IMAGES MS_ROW2   $RESULTS_PATH 7 33 5  50 1 0.01 1 False 2 1 True  False True  "${GPUS[@]}" # Table 4, Row 8
    train_KSDD $SAVE_IMAGES MS_ROW3   $RESULTS_PATH 7 33 5  50 1 0.01 1 False 2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 9
    train_KSDD $SAVE_IMAGES MS_ROW4   $RESULTS_PATH 7 33 5  50 1 0.01 1 True  2 1 True  False True  "${GPUS[@]}" # Table 4, Row 10
    train_KSDD $SAVE_IMAGES MS_ROW5   $RESULTS_PATH 7 33 5  50 1 0.01 1 True  2 1 False True  True  "${GPUS[@]}" # Table 4, Row 11
    # MS_ROW6 is N_5 from above

    # WS_ROW1 is N_0 from above
    train_KSDD $SAVE_IMAGES WS_ROW2   $RESULTS_PATH 7 33 0  50 1 0.01 1 True  2 1 True True  True  "${GPUS[@]}" # Table 4, Row 14


    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW1  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 False 2 1 False False True  "${GPUS[@]}" # Table 4, Row 1
    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW2  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 False 2 1 True  False True  "${GPUS[@]}" # Table 4, Row 2
    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW3  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 False 2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 3
    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW4  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 True  2 1 True  False True  "${GPUS[@]}" # Table 4, Row 4
    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW5  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 True  2 1 False True  True  "${GPUS[@]}" # Table 4, Row 5
    train_single $SAVE_IMAGES STEEL $STEEL_PATH FS_ROW6  $RESULTS_PATH 1 1000 1000 50 0.1 0.1 10 True  2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 6

    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW1  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 False 2 1 False False True  "${GPUS[@]}" # Table 4, Row 7
    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW2  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 False 2 1 True  False True  "${GPUS[@]}" # Table 4, Row 8
    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW3  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 False 2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 9
    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW4  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 True  2 1 True  False True  "${GPUS[@]}" # Table 4, Row 10
    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW5  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 True  2 1 False True  True  "${GPUS[@]}" # Table 4, Row 11
    train_single $SAVE_IMAGES STEEL $STEEL_PATH MS_ROW6  $RESULTS_PATH 1 1000 250 50 0.1 0.1 10 True  2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 12

    train_single $SAVE_IMAGES STEEL $STEEL_PATH WS_ROW1  $RESULTS_PATH 1 1000 0 50 0.1 0.1 10 True  2 1 False True  True  "${GPUS[@]}" # Table 4, Row 13
    train_single $SAVE_IMAGES STEEL $STEEL_PATH WS_ROW2  $RESULTS_PATH 1 1000 0 50 0.1 0.1 10 True  2 1 True  True  True  "${GPUS[@]}" # Table 4, Row 14




}


# Space delimited list of GPU IDs which will be used for training
GPUS=(5 6 7)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi

run_COMIND_EXPERIMENTS ./results-comind True "${GPUS[@]}"


