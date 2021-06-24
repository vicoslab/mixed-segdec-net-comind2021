# Mixed supervision for surface-defect detection: from weakly to fully supervised learning [Computers in Industry 2021]
Official PyTorch implementation for ["Mixed supervision for surface-defect detection: from weakly to fully supervised learning"](http://prints.vicos.si/publications/385) published in journal Computers in Industry 2021.

The same code is also an offical implementation of the method used in ["End-to-end training of a two-stage neural network for defect detection"](http://prints.vicos.si/publications/383) published in International Conference on Pattern Recognition 2020.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] 

Code and the dataset are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. For comerical use please contact danijel.skocaj@fri.uni-lj.si.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


## Citation
Please cite our [Computers in Industry 2021 paper](http://prints.vicos.si/publications/385) when using this code:

```
@article{Bozic2021COMIND,
  author = {Bo{\v{z}}i{\v{c}}, Jakob and Tabernik, Domen and 
  Sko{\v{c}}aj, Danijel},
  journal = {Computers in Industry},
  title = {{Mixed supervision for surface-defect detection: from weakly to fully supervised learning}},
  year = {2021}
}
```

## How to run:

### Requirements
Code has been tested to work on:
+ Python 3.8
+ PyTorch 1.6, 1.8
+ CUDA 10.0, 10.1
+ using additional packages as listed in requirements.txt

### Datasets
You will need to download the datasets yourself. For DAGM and Severstal Steel Defect Dataset you will also need a Kaggle account.
* DAGM available [here.](https://www.kaggle.com/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* KolektorSDD available [here.](https://www.vicos.si/Downloads/KolektorSDD)
* KolektorSDD2 available [here.](https://www.vicos.si/Downloads/KolektorSDD2)
* Severstal Steel Defect Dataset available [here.](https://www.kaggle.com/c/severstal-steel-defect-detection/data)

For details about data structure refer to `README.md` in `datasets` folder.

Cross-validation splits, train/test splits and weakly/fully labeled splits for all datasets are located in `splits` directory of this repository, alongside the instructions on how to use them.

##### Using on other data

Refer to `README.md` in `datasets` for instructions on how to use the method on other datasets. 

### Demo - fully supervised learning

To run fully supervised learning and evaluation on all four datasets run:

```bash
./DEMO.sh
# or by specifying multiple GPU ids 
./DEMO.sh 0 1 2
```
Results will be written to `./results` folder.

### Replicating paper results

To replicate the results published in the paper run:
```bash
./EXPERIMENTS_COMIND.sh
# or by specifying multiple GPU ids 
./EXPERIMENTS_COMIND.sh 0 1 2
```
To replicate the results from [ICPR 2020 paper](http://prints.vicos.si/publications/383): 
```
@misc{Bozic2020ICPR,
    title={End-to-end training of a two-stage neural network for defect detection},
    author={Jakob Božič and Domen Tabernik and Danijel Skočaj},
    year={2020},
    eprint={2007.07676},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
run:
```bash
./EXPERIMENTS_ICPR.sh
# or by specifying multiple GPU ids 
./EXPERIMENTS_ICPR.sh 0 1 2

```

Results will be written to `./results-comind` and `./results-icpr` folders.

### Usage of training/evaluation code
The following python files are used to train/evaluate the model:
+ `train_net.py` Main entry for training and evaluation
+ `models.py` Model file for network
+ `data/dataset_catalog.py` Contains currently supported datasets

In order to train and evaluate a network you can also use `EXPERIMENTS_ROOT.sh`, which contains several functions that will make training and evaluation easier for you.
For more details see the file `EXPERIMENTS_ROOT.sh`.  

#### Running code
Simplest way to train and evaluate a network is to use `EXPERIMENTS_ROOT.sh`, you can see examples of use in `EXPERIMENTS_ICPR.sh` and in `EXPERIMENTS_COMIND.sh`

If you wish to do it the other way you can do it by running `train_net.py` and passing the parameters as keyword arguments.
Bellow is an example of how to train a model for a single fold of `KSDD` dataset.

    python -u train_net.py  \
        --GPU=0 \
        --DATASET=KSDD \
        --RUN_NAME=RUN_NAME \
        --DATASET_PATH=/path/to/dataset \
        --RESULTS_PATH=/path/to/save/results \
        --SAVE_IMAGES=True \
        --DILATE=7 \
        --EPOCHS=50 \
        --LEARNING_RATE=1.0 \
        --DELTA_CLS_LOSS=0.01 \
        --BATCH_SIZE=1 \
        --WEIGHTED_SEG_LOSS=True \
        --WEIGHTED_SEG_LOSS_P=2 \
        --WEIGHTED_SEG_LOSS_MAX=1 \
        --DYN_BALANCED_LOSS=True \
        --GRADIENT_ADJUSTMENT=True \
        --FREQUENCY_SAMPLING=True \
        --TRAIN_NUM=33 \
        --NUM_SEGMENTED=33 \
        --FOLD=0

Some of the datasets do not require you to specify `--TRAIN_NUM` or `--FOLD`-
After training, each model is also evaluated.

For KSDD you need to combine the results of evaluation from all three folds, you can do this by using `join_folds_results.py`:

    python -u join_folds_results.py \
        --RUN_NAME=SAMPLE_RUN \
        --RESULTS_PATH=/path/to/save/results \
        --DATASET=KSDD 
        
You can use `read_results.py` to generate a table of results f0r all runs for selected dataset.        
Note: The model is sensitive to random initialization and data shuffles during the training and will lead to different performance with different runs unless `--REPRODUCIBLE_RUN` is set.        

