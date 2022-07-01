
## Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation
![](resources/fig1.png)
By [Lin Chen*](https://scholar.google.com/citations?user=-t92FH8AAAAJ&hl=zh-CN&oi=sra)

## Overview 
This repo is the official PyTorch implementation of Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation.
![](resources/fig2.png)

The code is based on mmsegmentaion.

## Environment
In this project, we use python 3.8.13 and pytorch==1.8.1, torchvision==0.9.1, mmcv-full==1.4.7, mmseg==0.22.1. 

The environment can be quickly set up as follows:

```shell
cd DDB
conda create -n ddb python=3.8
conda activate ddb
make install
```
## Results
 | checkpoints                                    | source           | target |mIoU |
 |------------------|--------|------|-------|
 | weights/gta+syn2cs/r2-ckd-pro-bs1x4/weight.pth | gta              | cityscape | 62.7|
 | weights/gta+syn2cs/r2-ckd-pro-bs1x4/weight.pth | gta<br>synscapes | cityscape | 69.0|
 | weights/gta2cs+map/r2-ckd-pro-bs1x4/weight.pth | gta              | cityscape<br>mapillary | 60.4<br>56.9|

The above checkpoints can be obtained through [GoogleDrive](https://drive.google.com/drive/folders/1mdIcSrOPTgzbB8zSFtoM2pzWlB0S3mpA?usp=sharing). 
After downloading, please put them under the project folder
## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**mapillary** Please, download MAPILLARY v1.2 from [here](https://research.mapillary.com/)  
**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.
them to `data/gta`.
Then, you should prepare data as follows:
```shell
cd DASS
mkdir data
# If you prepare the data at the first time, you should convert the data for validation
python tools/convert_datasets/gta.py data/gta/ # Source domain
python tools/convert_datasets/synthia.py data/synthia/ # Source domain
python tools/convert_datasets/synscapes.py data/synscapes/ # Source domain
# convert mapillary to cityscape format and resize it for efficient validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary/ \
data/mapillary/cityscape_trainIdLabel --train_id # Source domain
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images \
data/mapillary/cityscape_trainIdLabel/val/label data/mapillary/half/val_img \
data/mapillary/half/val_label
```

The final folder structure should look like this:

```none
DASS
├── ...
├── weights
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscape_trainIdLabel
│   │   ├── half
│   │   |   ├── val_img
│   │   |   ├── val_label
├── ...
```

## Evaluation
Download the folder  [weights](https://pan.baidu.com/s/1hP9cNI0qWGd78Clg77I55w?pwd=8i2j) and place it in the project directory
Verify by selecting the different config files in `configs/tests`
```shell
python tools/test.py {config} {weight} --eval mIoU
```

## Training

### Step 1

Using following commands, you will receive two complementary teacher models (cu_model and ca_model)
```shell
# Train on the region-path (using cut-mix for domain bridging)
python tools/train.py configs/gtav2cityscapes/r1_st_cu_dlv2_r101v1c_1x4_512x512_40k_gtav2cityscapes.py
# Train on the class-path (using class-mix for domain bridging)
python tools/train.py configs/gtav2cityscapes/r1_st_ca_dlv2_r101v1c_1x4_512x512_40k_gtav2cityscapes.py
# Train on the region-path (using cut-mix for domain bridging) (Train with multiple GPUs)
bash tools/dist_train.sh configs/gtav2cityscapes/r1_st_ca_dlv2_r101v1c_2x2_512x512_40k_gtav2cityscapes.py ${GPU_NUM}
# Train on the class-path (using class-mix for domain bridging) (Train with multiple GPUs)
bash tools/dist_train.sh configs/gtav2cityscapes/r1_st_cu_dlv2_r101v1c_2x2_512x512_40k_gtav2cityscapes.py ${GPU_NUM}
```

If you want to generate prototypes for rectifying the pseudo label produced in Step 2. You should run:
```shell
# Generating prototypes for the region-path teacher on target domain
python tools/cal_prototypes/cal_prototype.py {CU_MODEL_CONFIG_DIR} --checkpoint={CU_MODEL_CHECKPOINT_DIR}
# Generating prototypes for the region-path teacher on target domain
python tools/cal_prototypes/cal_prototype.py {CA_MODEL_CONFIG_DIR} --checkpoint={CA_MODEL_CHECKPOINT_DIR}
```

### Step 2

After step 1, you should rename the checkpoints and put them in the checkpoints' folder manually. Such as:

```none
DASS
├── ...
├── checkpoints
│   ├── gta2cs_stage1
│   │   ├── gta2cs_st-cu_dlv2.pth
│   │   ├── gta2cs_st-ca_dlv2.pth
├── ...
```

Then, you can run the following command for Cross-path Knowledge Aggregation:
```shell
# Distillate the knowledge from two teacher models to a student model
python tools/train.py configs/gtav2cityscapes/r1_ckd_dlv2_r101v1c_1x4_512x512_40k_gtav2cityscapes.py
# Train with multiple GPUs
bash tools/dist_train.sh configs/gtav2cityscapes/r1_ckd_dlv2_r101v1c_2x2_512x512_40k_gtav2cityscapes.py ${GPU_NUM}
```

### Step 1 on Round 2

```shell
# Self-training again with weights initialized by step2 on stage 1
python tools/train.py configs/uda/st/gta2cs_st-cu-r2_dlv2red-adapter_r101v1c_poly10warm_s0.py
# Self-training again with weights initialized by step2 on stage 1
python tools/train.py configs/uda/st/gta2cs_st-ca-r2_dlv2red-adapter_r101v1c_poly10warm_s0.py
```

### ...
