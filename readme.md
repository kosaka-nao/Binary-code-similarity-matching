# Binary-code-similarity-matching 

This repository contains the code we engineered for our research on "Binary Code similarity matching" for COMP5214 Course Project. 

# Remark and caustion

Becareful, it takes up 70GB ram to load the dataset for training.

# Dataset

You can download our pre-processed dataset from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wkwongal_connect_ust_hk/EpwO-aH7DdlNraeOb91W3OUBHSs-H6FZHNTHCpwOLChoqQ?e=Exy8DT)

## Dataset file overview
- ffmpeg_dataset.pickle: The baseline dataset of FFmpeg
- ffmpeg_dataset_wl_kernel.pickle: The dataset applied W-L graph test to pairwise
- utils_sample.zip: The compiled utilies binaries 
- ffmpeg_sample.zip: The compiled ffmpeg binaries 

Remark: The preprocessed Binutils, Coreutils, OpenSSL, and Zlib dataset were a bit large(15GB) and we experience issue on uploading to the onedrive, hence it isn't included in this artifact submission. 

# Usage

0. Please run the following to setup the python libraries required to run model:

```bash
pip3 install -r requirements.txt
```

1. Download all the dataset we provided in link.txt, unzip and paste it under the same folder running the python

2. To run the baseline case (Randomly pairwise training set, refer to section 4.4 of our report)

```bash
python3 train_function_similarity_baseline.py
```

3. To run the graph test case (Pairwise with low similarirty CFG pair, refer to section 4.4 of our report)

```bash
python3 train_function_similarity_WL.py
```

4. If you want to train on another dataset, please refer to ```disassemble_util.py``` and re-config accordingly 

5. You can use ```topN_evaluation.py``` to run the searching task mentioned on section 4.4. 
