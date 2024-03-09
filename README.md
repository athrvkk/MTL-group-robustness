# Multitask Learning Can Improve Worst-Group Outcomes

[![openreview](https://img.shields.io/badge/OpenReview-TMLR_02/2024-blue
)](https://openreview.net/forum?id=sPlhAIp6mk)


## Table of Contents
1. [Environment](#environment)
2. [Download, extract and Generate metadata for datasets](#download-extract-and-generate-metadata-for-datasets)
3. [Reproducing Paper Results](#reproducing-paper-results)
4. [Additional Support/Issues?](#additional-supportissues)
5. [Citation](#citation)


## Environment
We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the environment. Our Python version is <code>3.11.5</code>. To create the environment, run the following command:

```
conda env create -f environment.yml -n mtl-group-robustness-env
```

To activate the environment, run the following command:

```
conda activate mtl-group-robustness-env
```


## Download, Extract and Generate metadata for datasets

To downloads, extracts and formats the datasets as per the code, run the following script. This will store the data and metadata in the <code>data</code> folder. It already contains the <code>civilcomments-small</code> dataset.

```bash
python3 ./src/setup_datasets.py dataset_name --download --data_path data
```

## Reproducing Paper Results

The <code>[./src/hparams.yaml](https://github.com/atharvajk98/MTL-group-robustness/blob/main/src/hparams.yaml)</code> file includes the optimal hyperparameters for each method across all five datasets. To get started, execute the following command to generate Python scripts for training with the best hyperparameters.

```bash
python3 ./src/generate_hyper_search_scripts.py --dataset waterbirds --method erm_mt_l1
```

This will create a txt file in the <code>hparams_files</code> folder, containing the Python script for five seeds. It will also generate an executable bash file in the <code>scripts</code>  folder. To start training run the following command:
```bash
sbatch ./scripts/train_waterbirds_erm_mt_l1_hp.sh
```
This will store the best results as a json file for each run in the <code>models_params</code> folder.


## Additional Support/Issues?

If you face any issues in our code / reporducing our results raise a Github issue or contact Atharva Kulkarni (atharvak@cs.cmu.edu)


## Citation 

```
@article{
kulkarni2024multitask,
title={Multitask Learning Can Improve Worst-Group Outcomes},
author={Atharva Kulkarni and Lucio M. Dery and Amrith Setlur and Aditi Raghunathan and Ameet Talwalkar and Graham Neubig},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=sPlhAIp6mk},
note={}
}
```