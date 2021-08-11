### Datasets
You will need to download the datasets yourself. For DAGM and Severstal Steel Defect Dataset you will also need a Kaggle account.
* DAGM available [here.](https://www.kaggle.com/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* KolektorSDD available [here.](https://www.vicos.si/Downloads/KolektorSDD) Use version with boxes for annotations).
* KolektorSDD2 available [here.](https://www.vicos.si/Downloads/KolektorSDD2)
* Severstal Steel Defect Dataset available [here.](https://www.kaggle.com/c/severstal-steel-defect-detection/data)

#### DAGM
After downloading and extracting the dataset, the file structure should look like this:
    
    DAGM    
    ├── Class1
    │   ├── Test
    │   └── Train
    ├── Class2
    │   ├── Test
    │   └── Train
        ...
        
#### KSDD
Run `KSDD/download_and_extract.py` script to automatically download and extract the dataset into the folder:

```bash
cd datasets/KSDD
python download_and_extract.py
```

After downloading and extracting the dataset, the file structure should look like this:

    KSDD
    ├── kos01
    ├── kos02
    ├── kos03
        ...


#### KSDD2
Run `KSDD2/download_and_extract.py` script to automatically download and extract the dataset into the folder:

```bash
cd datasets/KSDD2
python download_and_extract.py
```

After downloading and extracting the dataset, the file structure should look like this:

    KSDD2
    ├── train
    ├── test


#### STEEL
After downloading and extracting the dataset, the file structure should look like this:

    STEEL
    ├── train_images
    ├── test_images
    ├── train.csv

Note that you do not need any images from kaggle test set and can delete test_images folder if you wish.    

### Using on other data

The easiest way to use the method on other datasets is to implement a new class that extends the `data.dataset.Dataset` class.
The only method you need to implement is `read_contents()`, in which you have to save the positive and negative images in 
`self.pos_imgs` and `self.neg_imgs`.
If your data can fit in the memory, you can use eager loading, where all of the images and labels are read at the beginning, 
or you can use on-demand-read, where you only initialize paths at the beginning and images and labels are then loaded when needed.
Each entry in `self.pos_imgs` and `self.neg_imgs` need to be of form `[image, segmentation_mask, segmentation_loss_mask, is_segmented, image_path, segmentation_mask_path, sample_name]`.
If you are using eager loading, you dont have to set `image_path` and `segmentation_mask_path`, and if you are using on-demand-loading, you
dont have to set `image`, `segmentation_mask` and `segmentation_mask_path`.
At the end of `read_contents()`, after you read the data, you also need to set `self.len`, `self.num_pos` and `self.num_neg` properties, 
and after that you need to call `self.init_extra()`.
For an example of eager loading you can check out `data.input_ksdd2.py`, and you can check out `data.input_steel.py` for an example of 
on-demand-load.

All of our implementations of datasets use `split_X_Y.pyb` files to get the images which are used for certain training scenarios, however if you wish to
use the code on your own data, there is no need to generate or use split files, all that matters is that you load the contents of the dataset as described above.

After you implemented data loading, you need to include the new dataset in `data.dataset_catalog.py` and
in `config.py` you need to set input dimensions.
