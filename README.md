# Select objects in pictures to delete and repair

## Introduction

This project uses Inpaint-Anything and gussion-grouping to first delete objects and repair backgrounds on old multi-angle photos, and then convert them into point cloud format files.

## Installation




Requires  `python>=3.8`


```python
!git clone https://github.com/geekyutao/Inpaint-Anything.git
```

    Cloning into 'Inpaint-Anything'
    


```python
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install -r lama/requirements.txt
```

If problems with `torchtext` problem during runtime, please uninstall it manually and install version 0.50


```python
!pip uninstall -y torchtext
!pip install torchtext==0.5.0
```

If you want to try more, please refer to[Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything?tab=readme-ov-file)

## Usage

Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [LaMa](./lama/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.


Run the code in the image_inpaint, please customize `input` and `output`

This code will traverse the entire `input folder`, delete and repair objects for each picture, and store them in the `output folder`

If you want to delete and repair objects in a single photo, please run the following code


```python
python remove_anything.py \
    --input_img you_image_path \
    --coords_type key_in \
    --point_coords 200 450 \  #Please change the point position to the pixel point of the object you want to delete.
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

# 2. Point cloud generation in gaussian-grouping


The installation and file compilation in the master_class_gs file are based on colab

The following is the environment installation based on conda package
For details, please refer to [gaussian-grouping](https://github.com/lkeab/gaussian-grouping/tree/main)

## Installation

Clone the repository locally


```python
git clone https://github.com/lkeab/gaussian-grouping.git
cd gaussian-grouping
```


```python
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

If you want to prepare the mask on your own data set, you also need to prepare the [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) environment


```python
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

## Prepare sam mask


1. The structure of a custom dataset

If you want to prepare masks on your own dataset, you will need the [DEVA](https://github.com/lkeab/gaussian-grouping/blob/main/Tracking-Anything-with-DEVA/README.md) python environment and checkpoints.


```python
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```

The first step is to use colmap to convert the initial camera pose and point cloud


```python
python convert.py -s <location>
```

Then, convert the SAM associated object mask.


```python
bash script/prepare_pseudo_label.sh input 1
```

## 2.Training and Rendering

For Gaussian grouping training and segmentation rendering of a trained 3D Gaussian grouping model


```python
bash script/train.sh input 1
```

### Remark

In the master_class_gs file, the colab based environment installation has been written and I have run it successfully without any issues with colmap and installing the submodules folder.
If you are installing based on conda package, please refer to [gussion-grouping](https://github.com/lkeab/gaussian-grouping/tree/main)

